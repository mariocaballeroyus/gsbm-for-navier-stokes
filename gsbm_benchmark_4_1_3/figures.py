"""
The figures of the colliding circles, from the outputs of sweep.py and
MainKratos.py in out/:

- out/gsbm_circles_subscale.png: for the symmetric Nitsche method with the
  shifted subscale (SUBSCALE_CASE), the relative L2 error against the position
  d of the moving circle at the degrees 1 to 5 on N = 100, the topology change
  at d = 0 dashed, and at p = 1 the errors of the WSBM, SBM and O-SBM weights
  of the GSBM paper, from its code (config/reference_errors_p1.csv, divided
  by ||u||), as points;
- out/gsbm_circles_refinement.png: the relative L2 error against d at the
  degrees 1 to 3, a panel per degree and a curve per mesh of REFINEMENT_COLOURS
  with all its positions, and below each panel the convergence in h, from
  out/refinement_N<N>_p<p>.csv and, for N = 100, out/sweep_p<p>.csv;
- out/gsbm_circles_animation.gif: the field at every d of skins.py, as the
  circles move, beside the errors with the current d marked;
- out/gsbm_circles_degrees.gif: the fields at the degrees 1 to 4 at every d,
  side by side on one color scale, the ghost regions as the circles move.

    python3 figures.py
"""

import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Text large enough to read in the README, which describes the figures
plt.rcParams.update({"font.size": 17, "axes.labelsize": 20, "axes.titlesize": 21, "xtick.labelsize": 17,
                     "ytick.labelsize": 17, "legend.fontsize": 16, "legend.title_fontsize": 16})

CASE = "circles_skew"
DEGREES = (1, 2, 3, 4, 5)
# One hue per degree, from the colour-blind-safe palette of Okabe and Ito
# (its black left to the paper's points), as in 4.1.1
COLOURS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7", 5: "#E69F00"}

# The symmetric Nitsche method with the shifted subscale, in a figure of its
# own
SUBSCALE_CASE = "circles_subscale"

# The meshes and degrees of the refinement study, whose step of d is never a
# whole number of knot spans, and the colour of each mesh in its figure
REFINEMENT_MESHES = (50, 100, 200, 325)
REFINEMENT_DEGREES = (1, 2, 3)
REFINEMENT_COLOURS = {50: "#E69F00", 100: "#0072B2", 200: "#CC79A7", 325: "#1f1f1e"}

# The knot spans per direction of the sweep and its fields
SWEEP_CELLS = 100

# The degree of the fields of the animation beside the errors, and the
# degrees of the animation of the ghost regions
FIELD_DEGREE = 5
GHOST_DEGREES = (1, 2, 3, 4)

# The errors of the GSBM paper at p = 1, from its code (absolute, Eq. 48), as
# points with a marker per weighting
REFERENCE = Path("config/reference_errors_p1.csv")
REFERENCE_WEIGHTS = (("standard", "WSBM", "s"), ("binary", "SBM", "^"), ("fraction", "O-SBM", "D"))


def read_fields(path):
    """Grid points and fields, indexed [x, y], of a legacy VTK structured points file."""
    header, *blocks = path.read_text().split("SCALARS ")
    keys = {line.split()[0]: line.split()[1:] for line in header.splitlines() if line.strip()}
    nx, ny = (int(n) for n in keys["DIMENSIONS"][:2])
    (x0, y0), (dx, dy) = ([float(v) for v in keys[key][:2]] for key in ("ORIGIN", "SPACING"))
    fields = {"x": x0 + dx * np.arange(nx), "y": y0 + dy * np.arange(ny)}
    for block in blocks:
        name, _, values = block.split("\n", 2)
        # VTK orders the points with x fastest
        fields[name.split()[0]] = np.array(values.split(), dtype=float).reshape(ny, nx).T
    return fields


def read_skin(path):
    """Closed skin polygons, one per line, of a legacy VTK polydata file."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("POINTS"))
    n = int(lines[start].split()[1])
    points = np.array([line.split()[:2] for line in lines[start + 1:start + 1 + n]], dtype=float)
    first = next(i for i, line in enumerate(lines) if line.startswith("LINES"))
    count = int(lines[first].split()[1])
    return [points[[int(i) for i in line.split()[1:]]] for line in lines[first + 1:first + 1 + count]]


def read_errors(column, case=CASE):
    """A column of out/sweep_p<p>.csv for a case, keyed by degree and position."""
    errors = {}
    for path in Path("out").glob("sweep_p*.csv"):
        degree = int(path.stem.removeprefix("sweep_p"))
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["case"] == case:
                    errors[(degree, float(row["d"]))] = float(row[column])
    return errors


def read_reference():
    """The paper's errors as relative ones, keyed by weighting and position:
    divided by the norm of u of our p = 1 runs at the same position."""
    absolute, relative = read_errors("l2_error"), read_errors("relative_error")
    norms = {d: absolute[(1, d)] / relative[(1, d)] for degree, d in absolute if degree == 1}
    with open(REFERENCE) as f:
        return {(row["weights"], float(row["d"])): float(row["l2norm"]) / norms[float(row["d"])]
                for row in csv.DictReader(f) if float(row["d"]) in norms}


def draw_sweep(axis, case=CASE):
    """The errors of a case against d at every degree, the paper's beside
    them, and the legends to the right."""
    axis.axvline(0., color="black", linestyle="--", linewidth=1.3)
    errors = read_errors("relative_error", case)
    for degree in DEGREES:
        pairs = sorted((d, error) for (p, d), error in errors.items() if p == degree)
        if pairs:
            positions, values = map(np.array, zip(*pairs))
            axis.semilogy(positions, values, color=COLOURS[degree], marker="o", markersize=6, linewidth=2.2,
                          label=f"$p = {degree}$")
    legends = [axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.))]

    if REFERENCE.exists():
        reference = read_reference()
        for weights, label, marker in REFERENCE_WEIGHTS:
            pairs = sorted((d, error) for (w, d), error in reference.items() if w == weights)
            positions, values = map(np.array, zip(*pairs))
            axis.semilogy(positions, values, linestyle="none", marker=marker, markersize=9, markerfacecolor="none",
                          markeredgecolor="black", markeredgewidth=1.6, label=label)
        handles, labels = axis.get_legend_handles_labels()
        legends.append(axis.legend(handles[-len(REFERENCE_WEIGHTS):], labels[-len(REFERENCE_WEIGHTS):],
                                   loc="lower left", bbox_to_anchor=(1.02, 0.), title="$p = 1$"))
    for legend in legends[:-1]:
        axis.add_artist(legend)

    axis.set_xlim(-0.21, 0.21)
    axis.set_xticks(np.linspace(-0.2, 0.2, 9))
    axis.set_xlabel("$d$")
    axis.set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    axis.grid(True, which="both", alpha=.3)
    return legends


def plot_sweep(case, path, title=None):
    figure, axis = plt.subplots(figsize=(13, 8), layout="constrained")
    legends = draw_sweep(axis, case)
    if title:
        axis.set_title(title)
    # The legends beside the axis, which the tight box would otherwise miss
    figure.savefig(path, dpi=110, bbox_inches="tight", pad_inches=.1,
                   bbox_extra_artists=legends)
    plt.close(figure)


def read_refinement():
    """The relative errors of the refinement study, keyed by mesh, degree and
    position, N = 100 from the sweep."""
    errors = {}
    for cells in REFINEMENT_MESHES:
        for degree in REFINEMENT_DEGREES:
            path = Path("out") / (f"sweep_p{degree}.csv" if cells == 100 else f"refinement_N{cells}_p{degree}.csv")
            if path.exists():
                with open(path) as f:
                    for row in csv.DictReader(f):
                        if row["case"] == CASE:
                            errors[(cells, degree, float(row["d"]))] = float(row["relative_error"])
    return errors


def plot_refinement():
    """The errors against d, a panel per degree, and below each its convergence
    in h over the meshes with all their positions: the median over d, a marker
    on every mesh, the range over d as error bars and the optimal slope dashed
    with its triangle."""
    errors = read_refinement()
    positions = sorted({d for _, _, d in errors})
    figure = plt.figure(figsize=(19, 11.5), layout="constrained")
    grid = figure.add_gridspec(2, len(REFINEMENT_DEGREES), height_ratios=(1., .5))
    sweeps = [figure.add_subplot(grid[0, i]) for i in range(len(REFINEMENT_DEGREES))]
    rates = [figure.add_subplot(grid[1, i]) for i in range(len(REFINEMENT_DEGREES))]
    for axis, rate, degree in zip(sweeps, rates, REFINEMENT_DEGREES):
        axis.axvline(0., color="black", linestyle="--", linewidth=1.3)
        meshes = [cells for cells in REFINEMENT_COLOURS if all((cells, degree, d) in errors for d in positions)]
        # A run in progress with the positions it has, the convergence below without it
        for cells, colour in REFINEMENT_COLOURS.items():
            done = [d for d in positions if (cells, degree, d) in errors]
            if done:
                axis.semilogy(done, [errors[(cells, degree, d)] for d in done], color=colour, marker="o", markersize=5,
                              linewidth=2., label=f"$N = {cells}$")
        axis.set_title(f"$p = {degree}$")
        axis.set_xlim(-0.21, 0.21)
        axis.set_xticks(np.linspace(-0.2, 0.2, 5))
        axis.set_xlabel("$d$")
        axis.grid(True, which="both", alpha=.3)
        axis.legend(loc="upper right")

        sizes = 1. / np.array(meshes, dtype=float)
        values = np.array([[errors[(cells, degree, d)] for d in positions] for cells in meshes])
        median = np.median(values, axis=1)
        rate.errorbar(sizes, median, yerr=(median - values.min(axis=1), values.max(axis=1) - median), color=COLOURS[degree],
                      marker="o", markersize=7, linewidth=2.2, elinewidth=1.6, capsize=6, capthick=1.6,
                      label="median and range over $d$")
        rate.set_xscale("log")
        rate.set_yscale("log")
        # The optimal slope below the range of the finest mesh, down to a triangle
        # just left of it, labelled with the slope on its left
        finest = 1. / max(REFINEMENT_COLOURS)
        anchor = values.min(axis=1)[-1] / 1.5 * (finest / sizes[-1]) ** (degree + 1)
        slope = lambda h: anchor * (h / finest) ** (degree + 1)
        left, right = finest / 1.3, finest / 1.3 * 1.2
        rate.loglog([left, 1. / min(REFINEMENT_COLOURS)], [slope(left), slope(1. / min(REFINEMENT_COLOURS))], "--",
                    color="#1f1f1e", linewidth=1.3)
        rate.plot([left, right, right, left], [slope(left), slope(left), slope(right), slope(left)], color="#1f1f1e",
                  linewidth=1.2)
        rate.text(left / 1.03, np.sqrt(slope(left) * slope(right)), f"${degree + 1}$", ha="right", va="center",
                  fontsize=15, color="#1f1f1e")
        rate.set_xlabel("$h$")
        rate.set_xlim(finest / 1.6, 1.15 / min(REFINEMENT_COLOURS))
        rate.set_xticks([1. / cells for cells in REFINEMENT_COLOURS], [f"$1/{cells}$" for cells in REFINEMENT_COLOURS])
        rate.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        rate.grid(True, which="both", alpha=.3)
        rate.legend(loc="lower right")
    # The sweeps on one scale, with room above the curves for the legends
    for axis in sweeps:
        axis.set_ylim(min(errors.values()) / 1.5, 8. * max(errors.values()))
    for axis in sweeps[1:] + rates[1:]:
        axis.tick_params(which="both", labelleft=False)
    for axis in rates:
        axis.set_ylim(min(errors.values()) / 40., 2. * max(errors.values()))
    label = r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$"
    sweeps[0].set_ylabel(label)
    rates[0].set_ylabel(label)
    figure.savefig("out/gsbm_circles_refinement.png", dpi=110, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def draw_skin(axis, skin):
    """The loops of a skin, black on white to show on any color."""
    for loop in skin:
        axis.plot(loop[:, 0], loop[:, 1], color="white", linewidth=3.)
        axis.plot(loop[:, 0], loop[:, 1], color="black", linewidth=1.4)


def draw_field(axis, field, skin, bound, title=None):
    """u_h over the whole unit square, its extension outside the skin
    included, linear in [-1, 1], the range of u, and logarithmic beyond, up
    to a bound."""
    norm = matplotlib.colors.SymLogNorm(linthresh=1., linscale=3., vmin=-bound, vmax=bound, base=10.)
    extent = (field["x"][0], field["x"][-1], field["y"][0], field["y"][-1])
    image = axis.imshow(field["u_h"].T, origin="lower", extent=extent, cmap="turbo", norm=norm,
                        interpolation="bilinear")
    draw_skin(axis, skin)
    axis.set_xlim(extent[:2])
    axis.set_ylim(extent[2:])
    axis.set_aspect("equal")
    axis.set_xlabel("$x$")
    axis.set_ylabel("$y$")
    if title:
        axis.set_title(title)
    return image


def field_colorbar(figure, image, axes, bound):
    """A color bar for fields drawn up to a bound, with ticks at 0, +-1 and
    the decades beyond."""
    decades = [10.**k for k in range(1, int(np.log10(bound)) + 1)]
    ticks = [-d for d in reversed(decades)] + [-1., 0., 1.] + decades
    labels = ([f"$-10^{{{k}}}$" for k in range(len(decades), 0, -1)] + ["$-1$", "$0$", "$1$"]
              + [f"$10^{{{k}}}$" for k in range(1, len(decades) + 1)])
    bar = figure.colorbar(image, ax=axes, shrink=.8, pad=.02, ticks=ticks)
    bar.set_ticklabels(labels)
    bar.ax.set_title("$u_h$", pad=14)


def animate():
    """The field at p = FIELD_DEGREE at every d, on one color scale, beside
    the errors with the current d marked."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from skins import D_VALUES, position_name
    names = [position_name(d) for d in D_VALUES]
    fields = [read_fields(Path(f"out/{CASE}_p{FIELD_DEGREE}_d{name}.vtk")) for name in names]
    skins = [read_skin(Path(f"out/skin_circles_nurbs_d{name}.vtk")) for name in names]
    bound = max(np.abs(field["u_h"]).max() for field in fields)

    # Room on the right for the legends of the sweep, which a frame does not
    # grow to fit
    figure = plt.figure(figsize=(23, 8.5), layout="constrained")
    figure.get_layout_engine().set(rect=(0., 0., .89, 1.))
    grid = figure.add_gridspec(1, 2, width_ratios=(1., 1.05))
    field_axis, sweep_axis = figure.add_subplot(grid[0, 0]), figure.add_subplot(grid[0, 1])
    image = draw_field(field_axis, fields[0], skins[0], bound)
    field_colorbar(figure, image, field_axis, bound)
    draw_sweep(sweep_axis)
    marker = sweep_axis.axvline(D_VALUES[0], color="0.3", linewidth=6., alpha=.35)

    def update(frame):
        image.set_data(fields[frame]["u_h"].T)
        for line in field_axis.lines[:]:
            line.remove()
        draw_skin(field_axis, skins[frame])
        field_axis.set_title(f"$p = {FIELD_DEGREE}$,  $N = {SWEEP_CELLS}$,  $d = {position_name(D_VALUES[frame])}$")
        marker.set_xdata([D_VALUES[frame]] * 2)

    # The circles approaching and merging, and back, to loop smoothly
    count = len(D_VALUES)
    frames = list(range(count - 1, -1, -1)) + list(range(1, count - 1))
    animation = FuncAnimation(figure, update, frames=frames)
    animation.save("out/gsbm_circles_animation.gif", writer=PillowWriter(fps=4), dpi=60)
    plt.close(figure)


def animate_degrees():
    """The fields at the degrees GHOST_DEGREES at every d, side by side on one
    color scale, each titled with its degree, as the circles move."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    from skins import D_VALUES, position_name
    names = [position_name(d) for d in D_VALUES]
    fields = {degree: [read_fields(Path(f"out/{CASE}_p{degree}_d{name}.vtk")) for name in names]
              for degree in GHOST_DEGREES}
    skins = [read_skin(Path(f"out/skin_circles_nurbs_d{name}.vtk")) for name in names]
    bound = max(np.abs(field["u_h"]).max() for frames in fields.values() for field in frames)

    figure, axes = plt.subplots(1, len(GHOST_DEGREES), figsize=(24, 7.2), layout="constrained")
    images = [draw_field(axis, fields[degree][0], skins[0], bound, f"$p = {degree}$")
              for axis, degree in zip(axes, GHOST_DEGREES)]
    field_colorbar(figure, images[0], axes, bound)
    title = figure.suptitle("")

    def update(frame):
        for axis, image, degree in zip(axes, images, GHOST_DEGREES):
            image.set_data(fields[degree][frame]["u_h"].T)
            for line in axis.lines[:]:
                line.remove()
            draw_skin(axis, skins[frame])
        title.set_text(f"$d = {position_name(D_VALUES[frame])}$")

    # The circles approaching and merging, and back, to loop smoothly
    count = len(D_VALUES)
    frames = list(range(count - 1, -1, -1)) + list(range(1, count - 1))
    animation = FuncAnimation(figure, update, frames=frames)
    animation.save("out/gsbm_circles_degrees.gif", writer=PillowWriter(fps=4), dpi=60)
    plt.close(figure)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    plot_sweep(SUBSCALE_CASE, "out/gsbm_circles_subscale.png", f"$N = {SWEEP_CELLS}$")
    plot_refinement()
    animate()
    animate_degrees()
