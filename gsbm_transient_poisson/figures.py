"""
The figures of the transient Poisson benchmark, in out/:

- out/gsbm_transient_poisson_solutions.gif: the manufactured solutions of the
  cases of config/ side by side, the moving circle that is the domain, the field
  faded outside it;
- out/gsbm_transient_poisson_<case>_fields.gif: from the fields of MainKratos.py
  with --fields in FIELDS_FOLDER, u_h over the background, its extension outside
  the circle included, beside the errors against time, the current time marked;
- out/gsbm_transient_poisson_sbm.png: the errors of the GSBM against those of
  the _sbm cases of config/, with binary weights;
- out/gsbm_transient_poisson_<case>_sbm_fields.gif: as the _fields one, for those
  cases, the errors of the GSBM faded behind;
- out/gsbm_transient_poisson_refinement_<case>.png: from the runs in
  REFINEMENT_FOLDER, the errors against time and the convergence in h.

    python3 figures.py
"""

import csv
import json
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import sympy as sp

plt.rcParams.update({"font.size": 15, "axes.labelsize": 18, "axes.titlesize": 18, "legend.fontsize": 14})

CASES = {"fixed": "Fixed checkerboard", "moving": "Moving checkerboard", "radial": "Radial"}
STEMS = {"fixed": "checkerboard_fixed", "moving": "checkerboard_moving", "radial": "radial"}
COLOURS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}

# The circle, its centre at the start and its speed along x, of the skin and of
# MoveSkinProcess in the json
RADIUS = 0.25
CENTER = (0.4, 0.5)
SPEED = 0.2
END_TIME = 1.0

# The fields of MainKratos.py with --fields, and their degree
FIELDS_FOLDER = Path("out/fields")
FIELD_DEGREE = 3

# The h-refinement: its folder, end time and time step, and its meshes, evenly
# spaced in log h, N = 100 from the first steps of the runs in out/
REFINEMENT_FOLDER = Path("out/refinement")
REFINEMENT_END_TIME = 0.25
REFINEMENT_TIME_STEP = 0.005
REFINEMENT_DEGREES = (1, 2, 3)
REFINEMENT_COLOURS = {50: "#E69F00", 71: "#56B4E9", 100: "#0072B2", 141: "#009E73", 200: "#CC79A7", 283: "#1f1f1e"}


def exact_solution(case):
    """The manufactured solution of a case, a function of x, y and t."""
    parameters = json.loads(Path(f"config/ProjectParameters_{STEMS[case]}.json").read_text())
    expression = next(process["Parameters"]["exact_solution"] for process in parameters["processes"]["source_process_list"])
    x, y, t = sp.symbols("x y t")
    return sp.lambdify((x, y, t), sp.sympify(expression.replace("^", "**"), locals={"x": x, "y": y, "t": t}), "numpy")


def animate_solutions():
    from matplotlib.animation import FuncAnimation, PillowWriter
    exact = {case: exact_solution(case) for case in CASES}
    samples = np.linspace(0., 1., 501)
    X, Y = np.meshgrid(samples, samples)
    theta = np.linspace(0., 2. * np.pi, 400)
    times = np.linspace(0., END_TIME, 41)

    def center(t):
        return CENTER[0] + SPEED * t, CENTER[1]

    def outside(t):
        cx, cy = center(t)
        return np.ma.masked_where((X - cx) ** 2 + (Y - cy) ** 2 <= RADIUS ** 2, np.ones_like(X))

    figure, axes = plt.subplots(1, len(CASES), figsize=(6. * len(CASES), 6.), sharey=True, layout="constrained")
    figure.get_layout_engine().set(w_pad=.2, h_pad=.15)
    images, fades, rings = {}, {}, {}
    for axis, (case, title) in zip(axes, CASES.items()):
        images[case] = axis.imshow(exact[case](X, Y, 0.), origin="lower", extent=(0, 1, 0, 1), cmap="RdBu_r", vmin=-1., vmax=1.)
        fades[case] = axis.imshow(outside(0.), origin="lower", extent=(0, 1, 0, 1), cmap="gray", vmin=0., vmax=1., alpha=.6)
        rings[case], = axis.plot(center(0.)[0] + RADIUS * np.cos(theta), center(0.)[1] + RADIUS * np.sin(theta), color="black",
                                 linewidth=2.)
        axis.set_title(title)
        axis.set_xlabel("$x$")
    axes[0].set_ylabel("$y$")
    figure.colorbar(images["fixed"], ax=axes, label="$u$", shrink=.9)
    time_label = figure.suptitle(f"$t = {times[0]:.3f}$")

    def update(frame):
        t = times[frame]
        for case in CASES:
            images[case].set_data(exact[case](X, Y, t))
            fades[case].set_data(outside(t))
            rings[case].set_xdata(center(t)[0] + RADIUS * np.cos(theta))
        time_label.set_text(f"$t = {t:.3f}$")

    # Forth and back, to loop smoothly
    frames = list(range(len(times))) + list(range(len(times) - 2, 0, -1))
    FuncAnimation(figure, update, frames=frames).save("out/gsbm_transient_poisson_solutions.gif",
                                                      writer=PillowWriter(fps=10), dpi=60)
    plt.close(figure)


def read_fields(path):
    """The fields, indexed [y, x], of a legacy VTK structured points file."""
    header, *blocks = path.read_text().split("SCALARS ")
    keys = {line.split()[0]: line.split()[1:] for line in header.splitlines() if line.strip()}
    nx, ny = (int(n) for n in keys["DIMENSIONS"][:2])
    fields = {}
    for block in blocks:
        name, _, values = block.split("\n", 2)
        fields[name.split()[0]] = np.array(values.split(), dtype=float).reshape(ny, nx)
    return fields


def animate_fields(case, histories, degree=FIELD_DEGREE, folder=FIELDS_FOLDER, path=None, suffix="", reference=None):
    """u_h, linear in [-1, 1], the range of u, and logarithmic beyond, beside the
    errors of the case against time at every degree, the current time marked:
    of the case with that suffix of its config, with the errors of a reference
    faded behind."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import SymLogNorm
    files = sorted(folder.glob(f"{STEMS[case]}{suffix}_p{degree}_t*.vtk"),
                   key=lambda file: float(file.stem.rsplit("_t", 1)[1]))
    if not files:
        return
    fields = [read_fields(file)["u_h"] for file in files]
    times = [float(file.stem.rsplit("_t", 1)[1]) for file in files]
    bound = max(1.01, max(np.abs(field).max() for field in fields))
    theta = np.linspace(0., 2. * np.pi, 400)

    figure, (field_axis, error_axis) = plt.subplots(1, 2, figsize=(15., 6.2), layout="constrained")
    image = field_axis.imshow(fields[0], origin="lower", extent=(0, 1, 0, 1), cmap="turbo",
                              norm=SymLogNorm(linthresh=1., linscale=3., vmin=-bound, vmax=bound, base=10.))
    figure.colorbar(image, ax=field_axis, label="$u_h$", shrink=.9)
    field_axis.set_xlabel("$x$")
    field_axis.set_ylabel("$y$")
    rings = [field_axis.plot([], [], color="white", linewidth=3.)[0], field_axis.plot([], [], color="black", linewidth=1.4)[0]]
    for (name, p, _, _), rows in sorted((reference or {}).items()):
        if name == case:
            error_axis.semilogy(rows[:, 0], rows[:, 1], color=COLOURS[p], linewidth=1., alpha=.35)
    for (name, p, cells, _), rows in sorted(histories.items()):
        if name == case:
            error_axis.semilogy(rows[:, 0], rows[:, 1], color=COLOURS[p], marker="o", markersize=2.5, linewidth=1.,
                                label=f"$p = {p}$, $N = {cells}$")
    error_axis.set_xlim(0., END_TIME)
    error_axis.set_xlabel("$t$")
    error_axis.set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    error_axis.grid(True, which="both", alpha=.3)
    # Room below the curves for the legend
    error_axis.set_ylim(bottom=error_axis.get_ylim()[0] / 8.)
    error_axis.legend(loc="lower right")
    marker = error_axis.axvline(times[0], color="0.3", linewidth=5., alpha=.35)

    def update(frame):
        t = times[frame]
        image.set_data(fields[frame])
        for ring in rings:
            ring.set_data(CENTER[0] + SPEED * t + RADIUS * np.cos(theta), CENTER[1] + RADIUS * np.sin(theta))
        method = "SBM, " if suffix == "_sbm" else ""
        field_axis.set_title(f"{method}{CASES[case]},  $p = {degree}$,  $t = {t:.3f}$")
        marker.set_xdata([t, t])

    order = list(range(len(files))) + list(range(len(files) - 2, 0, -1))
    FuncAnimation(figure, update, frames=order).save(path or f"out/gsbm_transient_poisson_{case}_fields.gif",
                                                     writer=PillowWriter(fps=8), dpi=60)
    plt.close(figure)


def read_histories(suffix=""):
    """The errors of every run in out/, and of those only in FIELDS_FOLDER, rows
    of time and relative error, keyed by case, degree, mesh and time step: of
    the cases with that suffix of their config, _sbm for binary weights."""
    histories = {}
    for case, stem in STEMS.items():
        for folder in (Path("out"), FIELDS_FOLDER):
            for path in folder.glob(f"{stem}{suffix}_p*_N*_dt*.csv"):
                match = re.fullmatch(rf"{stem}{suffix}_p(\d+)_N(\d+)_dt([\d.e-]+)", path.stem)
                if not match:
                    continue
                key = (case, int(match[1]), int(match[2]), float(match[3]))
                with open(path) as f:
                    rows = [(float(row["time"]), float(row["relative_error"])) for row in csv.DictReader(f)]
                if rows and key not in histories:
                    histories[key] = np.array(rows)
    return histories


def plot_sbm(histories, sbm_histories):
    """The errors of the GSBM, filled markers, and of the SBM, hollow and dashed,
    a panel per case and a colour per degree."""
    from matplotlib.lines import Line2D
    figure, axes = plt.subplots(1, len(CASES), figsize=(7. * len(CASES), 6.5), sharex=True, sharey=True,
                                layout="constrained")
    for axis, case in zip(axes, CASES):
        for (name, degree, _, _), rows in sorted(histories.items()):
            if name == case:
                axis.semilogy(rows[:, 0], rows[:, 1], color=COLOURS[degree], marker="o", markersize=2.5, linewidth=1.)
        for (name, degree, _, _), rows in sorted(sbm_histories.items()):
            if name == case:
                axis.semilogy(rows[:, 0], rows[:, 1], color=COLOURS[degree], marker="o", markersize=3.5,
                              markerfacecolor="none", markeredgewidth=.6, linewidth=.8, linestyle="--")
        axis.set_title(CASES[case])
        axis.set_xlabel("$t$")
        axis.grid(True, which="both", alpha=.3)
    axes[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    # Room below the curves for the legends
    axes[0].set_ylim(bottom=axes[0].get_ylim()[0] / 5.)
    degrees = sorted({key[1] for key in (*histories, *sbm_histories)})
    axes[0].legend(handles=[Line2D([], [], color=COLOURS[p], linewidth=2., label=f"$p = {p}$") for p in degrees],
                   loc="lower left")
    axes[1].legend(handles=[Line2D([], [], color="0.2", marker="o", markersize=4., linewidth=1., label="GSBM"),
                            Line2D([], [], color="0.2", marker="o", markersize=5., markerfacecolor="none",
                                   linewidth=1., linestyle="--", label="SBM (binary weights)")],
                   loc="lower left")
    figure.savefig("out/gsbm_transient_poisson_sbm.png", dpi=110, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def read_refinement(case):
    """The refinement study of a case, keyed by mesh and degree: rows of time, L2
    error, L2 norm of u and time step up to its end time."""
    runs = {}
    for cells in REFINEMENT_COLOURS:
        for degree in REFINEMENT_DEGREES:
            folder = Path("out") if cells == 100 else REFINEMENT_FOLDER
            path = folder / f"{STEMS[case]}_p{degree}_N{cells}_dt{REFINEMENT_TIME_STEP:g}.csv"
            if path.exists():
                with open(path) as f:
                    rows = [[float(row[name]) for name in ("time", "l2_error", "l2_norm", "dt")] for row in csv.DictReader(f)
                            if float(row["time"]) <= REFINEMENT_END_TIME + 1e-9]
                if rows:
                    runs[(cells, degree)] = np.array(rows)
    return runs


def time_norm(rows):
    """The relative L2(L2) error, Eq. (54) of the error over that of u."""
    return np.sqrt(np.sum(rows[:, 3] * rows[:, 1] ** 2) / np.sum(rows[:, 3] * rows[:, 2] ** 2))


def plot_refinement(case):
    """The errors of a case against time, a panel per degree, and below each its
    convergence in h over the meshes with all their steps: the L2(L2) norm, a
    marker on every mesh, the range over time as error bars, whose top is the
    L-infinity(L2) norm, and the optimal slope dashed with its triangle."""
    runs = read_refinement(case)
    if not runs:
        return
    all_steps = int(round(REFINEMENT_END_TIME / REFINEMENT_TIME_STEP))
    relative = {key: rows[:, 1] / rows[:, 2] for key, rows in runs.items()}
    # A run that blows up stays in its panel, leaving the scale, out of the convergence
    diverged = {key for key, values in relative.items() if values.max() > 1.}
    stable = [values for key, values in relative.items() if key not in diverged]
    low, high = min(values.min() for values in stable), max(values.max() for values in stable)
    figure = plt.figure(figsize=(19, 11.5), layout="constrained")
    grid = figure.add_gridspec(2, len(REFINEMENT_DEGREES), height_ratios=(1., .5))
    curves = [figure.add_subplot(grid[0, i]) for i in range(len(REFINEMENT_DEGREES))]
    rates = [figure.add_subplot(grid[1, i]) for i in range(len(REFINEMENT_DEGREES))]
    finest, coarsest = 1. / max(REFINEMENT_COLOURS), 1. / min(REFINEMENT_COLOURS)
    for axis, rate, degree in zip(curves, rates, REFINEMENT_DEGREES):
        # A run in progress with the steps it has, the convergence below without it
        for cells, colour in REFINEMENT_COLOURS.items():
            if (cells, degree) in runs:
                axis.semilogy(runs[(cells, degree)][:, 0], relative[(cells, degree)], color=colour, marker="o", markersize=4,
                              linewidth=1.6, label=f"$N = {cells}$" + (" (diverges)" if (cells, degree) in diverged else ""))
        axis.set_title(f"{CASES[case]},  $p = {degree}$")
        axis.set_xlim(0., REFINEMENT_END_TIME)
        axis.set_xlabel("$t$")
        axis.grid(True, which="both", alpha=.3)
        axis.legend(loc="upper right", ncols=2)

        meshes = [cells for cells in REFINEMENT_COLOURS if (cells, degree) in runs and len(runs[(cells, degree)]) >= all_steps
                  and (cells, degree) not in diverged]
        if meshes:
            sizes = 1. / np.array(meshes, dtype=float)
            norms = np.array([time_norm(runs[(cells, degree)]) for cells in meshes])
            smallest = np.array([relative[(cells, degree)].min() for cells in meshes])
            largest = np.array([relative[(cells, degree)].max() for cells in meshes])
            rate.errorbar(sizes, norms, yerr=(norms - smallest, largest - norms), color=COLOURS[degree], marker="o", markersize=7,
                          linewidth=2.2, elinewidth=1.6, capsize=6, capthick=1.6, label=r"$L^2(L^2)$, range over $t$")
            # The optimal slope below the range of the finest mesh, down to a triangle
            # just left of it, labelled with the slope on its left
            anchor = smallest[-1] / 1.5 * (finest / sizes[-1]) ** (degree + 1)
            slope = lambda h: anchor * (h / finest) ** (degree + 1)
            left, right = finest / 1.3, finest / 1.3 * 1.2
            rate.loglog([left, coarsest], [slope(left), slope(coarsest)], "--", color="#1f1f1e", linewidth=1.3)
            rate.plot([left, right, right, left], [slope(left), slope(left), slope(right), slope(left)], color="#1f1f1e",
                      linewidth=1.2)
            rate.text(left / 1.03, np.sqrt(slope(left) * slope(right)), f"${degree + 1}$", ha="right", va="center",
                      fontsize=15, color="#1f1f1e")
            rate.legend(loc="lower right")
        rate.set_xscale("log")
        rate.set_yscale("log")
        rate.set_xlabel("$h$")
        rate.set_xlim(finest / 1.6, 1.15 * coarsest)
        rate.set_xticks([1. / cells for cells in REFINEMENT_COLOURS], [f"$1/{cells}$" for cells in REFINEMENT_COLOURS])
        rate.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        rate.grid(True, which="both", alpha=.3)
    # On one scale, with room above the curves for the legends
    for axis in curves:
        axis.set_ylim(low / 1.5, 8. * high)
    for axis in rates:
        axis.set_ylim(low / 40., 2. * high)
    for axis in curves[1:] + rates[1:]:
        axis.tick_params(which="both", labelleft=False)
    curves[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    rates[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    figure.savefig(f"out/gsbm_transient_poisson_refinement_{case}.png", dpi=110, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    animate_solutions()
    for case in CASES:
        plot_refinement(case)
    histories = read_histories()
    for case in CASES:
        animate_fields(case, histories)
    sbm_histories = read_histories("_sbm")
    if sbm_histories:
        plot_sbm(histories, sbm_histories)
    for case in CASES:
        animate_fields(case, sbm_histories, path=f"out/gsbm_transient_poisson_{case}_sbm_fields.gif", suffix="_sbm",
                       reference=histories)
