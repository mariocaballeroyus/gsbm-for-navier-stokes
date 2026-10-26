"""
The figures of the colliding circles, from the outputs of sweep.py and
MainKratos.py in out/:

- out/gsbm_circles_sweep.png: the relative L2 error against the position d of
  the moving circle at the degrees 1 to 5 on N = 100, the topology change at
  d = 0 dashed, dashed too the standard Taylor shift without the shifted
  subscale (STANDARD_CASE), and at p = 1 the errors of the WSBM, SBM and
  O-SBM weights of the GSBM paper, from its code
  (config/reference_errors_p1.csv, divided by ||u||), as points;
- out/gsbm_circles_animation.gif: the field at every d of skins.py, as the
  circles move, beside the errors with the current d marked.

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

CASE = "circles_taylor"
DEGREES = (1, 2, 3, 4, 5)
COLOURS = {1: "C0", 2: "C1", 3: "C2", 4: "C3", 5: "C4"}

# The degree of the fields of the animation
FIELD_DEGREE = 5

# The standard Taylor shift without the shifted subscale, beside CASE
STANDARD_CASE = "circles_standard"

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


def draw_sweep(axis):
    """The errors against d at every degree, those without the subscale and
    the paper's beside them, and the legends to the right."""
    axis.axvline(0., color="black", linestyle="--", linewidth=1.3)
    for errors, style, label in ((read_errors("relative_error"), {"marker": "o", "markersize": 6, "linewidth": 2.2},
                                  True),
                                 (read_errors("relative_error", STANDARD_CASE),
                                  {"linestyle": "--", "marker": "s", "markersize": 5, "markerfacecolor": "none",
                                   "linewidth": 1.8}, False)):
        for degree in DEGREES:
            pairs = sorted((d, error) for (p, d), error in errors.items() if p == degree)
            if pairs:
                positions, values = map(np.array, zip(*pairs))
                axis.semilogy(positions, values, color=COLOURS[degree], label=f"$p = {degree}$" if label else None,
                              **style)
    legends = [axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.))]
    if read_errors("relative_error", STANDARD_CASE):
        line = matplotlib.lines.Line2D
        legends.append(axis.legend(handles=[
            line([], [], color="black", marker="o", markersize=6, linewidth=2.2, label=r"with $u'_h$"),
            line([], [], color="black", linestyle="--", marker="s", markersize=5, markerfacecolor="none",
                 linewidth=1.8, label=r"without $u'_h$")], loc="center left", bbox_to_anchor=(1.02, .5),
            handlelength=3.))

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


def plot_sweep():
    figure, axis = plt.subplots(figsize=(13, 8), layout="constrained")
    legends = draw_sweep(axis)
    # The legends beside the axis, which the tight box would otherwise miss
    figure.savefig("out/gsbm_circles_sweep.png", dpi=110, bbox_inches="tight", pad_inches=.1,
                   bbox_extra_artists=legends)
    plt.close(figure)


def draw_field(axis, field, skin, bound, title=None):
    """u_h over the whole unit square, its extension outside the skin
    included, linear in [-1, 1], the range of u, and logarithmic beyond, up
    to a bound."""
    norm = matplotlib.colors.SymLogNorm(linthresh=1., linscale=3., vmin=-bound, vmax=bound, base=10.)
    extent = (field["x"][0], field["x"][-1], field["y"][0], field["y"][-1])
    image = axis.imshow(field["u_h"].T, origin="lower", extent=extent, cmap="turbo", norm=norm,
                        interpolation="bilinear")
    for loop in skin:
        axis.plot(loop[:, 0], loop[:, 1], color="white", linewidth=3.)
        axis.plot(loop[:, 0], loop[:, 1], color="black", linewidth=1.4)
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
        for loop in skins[frame]:
            field_axis.plot(loop[:, 0], loop[:, 1], color="white", linewidth=3.)
            field_axis.plot(loop[:, 0], loop[:, 1], color="black", linewidth=1.4)
        field_axis.set_title(f"$p = {FIELD_DEGREE}$,  $d = {position_name(D_VALUES[frame])}$")
        marker.set_xdata([D_VALUES[frame]] * 2)

    # The circles approaching and merging, and back, to loop smoothly
    count = len(D_VALUES)
    frames = list(range(count - 1, -1, -1)) + list(range(1, count - 1))
    animation = FuncAnimation(figure, update, frames=frames)
    animation.save("out/gsbm_circles_animation.gif", writer=PillowWriter(fps=4), dpi=60)
    plt.close(figure)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    plot_sweep()
    animate()
