"""
The figures of the benchmark, as those of the IGUANA study, from the outputs
of MainKratos.py and convergence.py in out/:

- out/gsbm_poisson_degrees_<domain>.png: u_h over the whole unit square, its
  extension outside the domain included, at the degrees 1 to 5 on N = 40, on
  one color scale, linear in [-1, 1] and logarithmic beyond, each titled with
  its degree;
- out/gsbm_poisson_shifts_<domain>.png: on the left, the field at p = 5 on
  the 40 x 40 grid of the json; on the right, a column for every shift in
  SHIFTS with its convergence at the degrees 1 to 5, relative errors against
  h from N = 40, a marker on every mesh, with the reference slope h^(p+1)
  of every degree dashed over its finest factor of two in h, and the
  convergence of UNENHANCED_SHIFT, the same method with the shift of order
  p, faded beneath;
- out/gsbm_poisson_subscale.png: the same convergence for the symmetric
  Nitsche method with the shifted subscale, at the degrees 1 to 5, a panel
  per domain;
- out/gsbm_poisson_domain_error.png: the convergence of DOMAIN_ERROR_SHIFT
  in the L2 norm inside the domain, a panel per domain, with its
  alpha-weighted error of Eq. (48) faded beneath.

The main method is the skew-symmetric Nitsche method without penalty, with
the enhanced shift, of order 2p, of config/MaterialParameters_skew_p<p>.json; the
shift of order p is that of config/MaterialParameters_skew_unenhanced_p<p>.json; the
shifted subscale, the cases <domain>_subscale of out/convergence_p<p>.csv,
is the symmetric Nitsche method with the penalty chi (p + 1)^2, chi = 20,
and the enhanced shift rewritten with the equation from the closest boundary
point. All have the ghost penalty of Sticko and Kreiss, kappa_p = 0.5 /
((p!)^2 (2p + 1)):

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

DOMAINS = ("circle", "flower", "parallelogram")
DEGREES = (1, 2, 3, 4, 5)
# One hue per degree, from the colour-blind-safe palette of Okabe and Ito
# (its black left to the slopes and the paper's points), and the near black
# of the slopes, as in the IGUANA study's figures
COLOURS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73", 4: "#CC79A7", 5: "#E69F00"}
INK = "#1f1f1e"

# The degrees of the convergence plots, and of the shifted subscale's
CONVERGENCE_DEGREES = (1, 2, 3, 4, 5)
SUBSCALE_DEGREES = (1, 2, 3, 4, 5)

# The shift of the fields, the mesh of the fields at all degrees, and the
# degree and mesh of the field beside the convergence
FIELD_SHIFT = "skew"
DEGREES_CELLS = 40
FIELD = (5, 40)

# The convergence plots, a column per shift: from N = 40, on h from 1/450,
# which leaves room for the slopes' triangles left of the finest mesh, and
# the same method with the shift of order p faded beneath
MIN_CELLS = 40
H_LIMITS = (1. / 450., 10.**-1.5)
SHIFTS = (("skew", "Skew-symmetric Nitsche, enhanced shift"),)
UNENHANCED_SHIFT = "skew_unenhanced"
# The height, in axes coordinates, kept below the legend of the degrees for the
# legend of the two shifts
SHIFTS_LEGEND_HEIGHT = .1

# The symmetric Nitsche method with the shifted subscale, in its own figure
SUBSCALE_SHIFT = "subscale"

# The shift whose error inside the domain has its own figure
DOMAIN_ERROR_SHIFT = "skew"

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
    """Closed skin polygon of a legacy VTK polydata file."""
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("POINTS"))
    n = int(lines[start].split()[1])
    points = np.array([line.split()[:2] for line in lines[start + 1:start + 1 + n]], dtype=float)
    return np.vstack([points, points[:1]])


def read_errors(column="relative_error"):
    """Relative errors in out/convergence_p<p>.csv, alpha-weighted or in the
    column given, keyed by shift, domain, degree and mesh, where they are."""
    errors = {}
    for path in Path("out").glob("convergence_p*.csv"):
        degree = int(path.stem.removeprefix("convergence_p"))
        with open(path) as f:
            for row in csv.DictReader(f):
                domain, shift = row["case"].split("_", 1)
                if row.get(column):
                    errors[(shift, domain, degree, int(row["N"]))] = float(row[column])
    return errors


def read_reference():
    """The paper's errors as relative ones, keyed by weighting, domain and
    mesh: divided by the norm of u of our p = 1 runs on the same mesh."""
    with open("out/convergence_p1.csv") as f:
        norms = {(row["case"].split("_", 1)[0], int(row["N"])): float(row["l2_error"]) / float(row["relative_error"])
                 for row in csv.DictReader(f) if row["case"].endswith("_" + FIELD_SHIFT)}
    with open(REFERENCE) as f:
        return {(row["weights"], row["domain"], int(row["n"])): float(row["l2norm"]) / norms[(row["domain"], int(row["n"]))]
                for row in csv.DictReader(f) if (row["domain"], int(row["n"])) in norms}


def plot_reference(axis, reference, domain, anchor=(0., 1.)):
    """The paper's errors as points, in a legend of their own at the top
    left, its corner at an anchor in axes coordinates; none on the domains
    that are not the paper's."""
    if not any(d == domain for (_, d, _) in reference):
        return
    degrees = axis.get_legend()
    for weights, label, marker in REFERENCE_WEIGHTS:
        pairs = sorted((cells, error) for (w, d, cells), error in reference.items()
                       if (w, d) == (weights, domain) and cells >= MIN_CELLS)
        cells, values = map(np.array, zip(*pairs))
        axis.loglog(1. / cells, values, linestyle="none", marker=marker, markersize=9, markerfacecolor="none",
                    markeredgecolor="black", markeredgewidth=1.6, label=label)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(handles[-len(REFERENCE_WEIGHTS):], labels[-len(REFERENCE_WEIGHTS):], loc="upper left",
                bbox_to_anchor=anchor, title="$p = 1$")
    axis.add_artist(degrees)


def draw_field(axis, field, skin, cells, bound, title=None):
    """u_h over the whole unit square, its extension outside the skin
    included, linear in [-1, 1], the range of u, and logarithmic beyond, up
    to a bound, with the cells x cells knot spans of the background."""
    norm = matplotlib.colors.SymLogNorm(linthresh=1., linscale=3., vmin=-bound, vmax=bound, base=10.)
    extent = (field["x"][0], field["x"][-1], field["y"][0], field["y"][-1])
    image = axis.imshow(field["u_h"].T, origin="lower", extent=extent, cmap="turbo", norm=norm,
                        interpolation="bilinear")
    axis.vlines(np.linspace(*extent[:2], cells + 1), *extent[2:], colors="black", linewidths=.3, alpha=.4)
    axis.hlines(np.linspace(*extent[2:], cells + 1), *extent[:2], colors="black", linewidths=.3, alpha=.4)
    axis.plot(skin[:, 0], skin[:, 1], color="white", linewidth=3.)
    axis.plot(skin[:, 0], skin[:, 1], color="black", linewidth=1.4)
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


def plot_degrees(domain):
    """The fields at every degree."""
    skin = read_skin(Path(f"out/skin_{domain}_nurbs.vtk"))
    fields = [read_fields(Path(f"out/{domain}_{FIELD_SHIFT}_p{degree}_N{DEGREES_CELLS}.vtk"))
              for degree in DEGREES]
    bound = max(np.abs(field["u_h"]).max() for field in fields)
    figure, axes = plt.subplots(1, len(DEGREES), figsize=(6 * len(DEGREES), 6.8), layout="constrained")

    for axis, field, degree in zip(axes, fields, DEGREES):
        image = draw_field(axis, field, skin, DEGREES_CELLS, bound, f"$p = {degree}$")

    field_colorbar(figure, image, axes, bound)
    figure.savefig(f"out/gsbm_poisson_degrees_{domain}.png", dpi=100, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def plot_convergence(axis, errors, shift, domain, title, degrees=CONVERGENCE_DEGREES, faded=False):
    """The errors of a shift at every degree, a marker on every mesh; faded,
    thinner and without labels, beneath another shift."""
    style = dict(alpha=.5, markersize=3.5, linewidth=1.4) if faded else dict(markersize=4.5, linewidth=2.2)
    for degree in degrees:
        pairs = sorted((cells, error) for (s, d, p, cells), error in errors.items()
                       if (s, d, p) == (shift, domain, degree) and cells >= MIN_CELLS)
        if len(pairs) < 2:
            continue
        cells, values = map(np.array, zip(*pairs))
        axis.loglog(1. / cells, values, color=COLOURS[degree], marker="o",
                    label="_faded" if faded else f"$p = {degree}$", **style)
    if title:
        axis.set_title(title)
    axis.grid(True, which="both", alpha=.3)
    if not faded:
        axis.legend(loc="lower right")


def plot_slopes(axis, errors, shift, domain, limits, degrees=CONVERGENCE_DEGREES):
    """The optimal slope p + 1 of each degree as in the IGUANA study's
    figures: dashed over its finest factor of two in h, below every error
    there, from a small triangle just left of the finest mesh, h in [1/400,
    1.2/400], labelled p + 1 on its left. A triangle that would overlap one
    already drawn moves down with its line. Returns the lowest value the
    triangles reach."""
    lowest = limits[1][0]
    placed = []
    for degree in degrees:
        values = {cells: error for (s, d, p, cells), error in errors.items() if (s, d, p) == (shift, domain, degree)}
        if not values:
            continue
        slope, finest = degree + 1, max(values)

        # The line through the fine end lies below every error it spans
        anchor = min(values[n] * (n / finest)**slope for n in values if 2 * n >= finest) / 1.6
        left, right = 1. / 400., 1.2 / 400.
        while True:
            bottom = anchor * (left * finest)**slope / 1.18
            top = bottom * 1.2**slope
            extent = (np.log10(bottom), np.log10(top))
            clashes = [b for b in placed if extent[0] < b[1] + .1 and b[0] < extent[1] + .1]
            if not clashes:
                break
            anchor *= 10.**(min(b[0] for b in clashes) - .14 - extent[1])
        placed.append(extent)
        lowest = min(lowest, bottom / 2.)

        span = np.array([left, 2. / finest])
        axis.loglog(span, anchor * (span * finest)**slope, "--", color=INK, linewidth=1.3)
        axis.plot([left, right, right, left], [bottom, bottom, top, bottom], color=INK, linewidth=1.2)
        axis.annotate(f"{slope}", (left, np.sqrt(bottom * top)), textcoords="offset points", xytext=(-4, 0),
                      ha="right", va="center", fontsize=15, color=INK)
    return lowest


def plot_shifts(domain, errors):
    """The field on the left, and on the right a column of convergence for
    each shift, slimmer, side by side on one y-axis."""
    figure = plt.figure(figsize=(12. + 7. * len(SHIFTS), 9.5), layout="constrained")
    grid = figure.add_gridspec(1, 2, width_ratios=(1., .25 + .6 * len(SHIFTS)))
    field_axis = figure.add_subplot(grid[0, 0])
    columns = grid[0, 1].subgridspec(1, len(SHIFTS), wspace=.03)
    first_axis = figure.add_subplot(columns[0, 0])
    axes = [first_axis] + [figure.add_subplot(columns[0, i], sharex=first_axis, sharey=first_axis)
                           for i in range(1, len(SHIFTS))]

    degree, cells = FIELD
    field = read_fields(Path(f"out/{domain}_{FIELD_SHIFT}_p{degree}_N{cells}.vtk"))
    bound = np.abs(field["u_h"]).max()
    image = draw_field(field_axis, field, read_skin(Path(f"out/skin_{domain}_nurbs.vtk")), cells, bound)
    field_axis.legend(handles=[matplotlib.lines.Line2D([], [], linestyle="none")] * 2,
                      labels=[f"$p = {degree}$", f"${cells} \\times {cells}$"],
                      loc="upper left", handlelength=0, handletextpad=0)
    field_colorbar(figure, image, field_axis, bound)

    # Column titles only to tell several shifts apart; the degrees, and below
    # them the two shifts by their styles where the faded one has errors
    reference = read_reference()
    unenhanced = any((s, d) == (UNENHANCED_SHIFT, domain) for s, d, _, _ in errors)
    styles = [matplotlib.lines.Line2D([], [], color="black", marker="o", markersize=3.5, linewidth=1.4, alpha=.5),
              matplotlib.lines.Line2D([], [], color="black", marker="o", markersize=4.5, linewidth=2.2)]
    for axis, (shift, title) in zip(axes, SHIFTS):
        plot_convergence(axis, errors, UNENHANCED_SHIFT, domain, None, faded=True)
        plot_convergence(axis, errors, shift, domain, title if len(SHIFTS) > 1 else None)
        if unenhanced:
            axis.legend(loc="lower right", bbox_to_anchor=(1., SHIFTS_LEGEND_HEIGHT))
        plot_reference(axis, reference, domain)
        if unenhanced:
            axis.add_artist(matplotlib.legend.Legend(axis, styles, ["standard shift ($p$)", "enhanced shift ($2p$)"],
                                                     loc="lower right"))
        axis.set_xlabel("$h$")
    first_axis.set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    for axis in axes[1:]:
        axis.tick_params(which="both", labelleft=False)

    # The slopes once the shared limits of all panels are known, h from 1/450
    # to the half decade above 1/40
    first_axis.set_xlim(*H_LIMITS)
    limits = first_axis.get_xlim(), first_axis.get_ylim()
    lowest = min(plot_slopes(axis, errors, shift, domain, limits) for axis, (shift, _) in zip(axes, SHIFTS))
    first_axis.set_xlim(limits[0])
    first_axis.set_ylim(lowest, limits[1][1])

    # Powers of ten at half decades, which the slim columns fit
    exponents = (-2.5, -2., -1.5)
    first_axis.set_xticks([10.**k for k in exponents], [f"$10^{{{k:g}}}$" for k in exponents])
    first_axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    figure.savefig(f"out/gsbm_poisson_shifts_{domain}.png", dpi=110, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def plot_subscale(errors):
    """The convergence of the symmetric Nitsche method with the shifted
    subscale on every domain, side by side on one y-axis, each titled with
    its domain."""
    figure, axes = plt.subplots(1, len(DOMAINS), figsize=(6.5 * len(DOMAINS), 8.5), sharex=True, sharey=True,
                                layout="constrained")
    reference = read_reference()
    for axis, domain in zip(axes, DOMAINS):
        plot_convergence(axis, errors, SUBSCALE_SHIFT, domain, domain.capitalize(), SUBSCALE_DEGREES)
        plot_reference(axis, reference, domain)
        axis.set_xlabel("$h$")
    axes[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    for axis in axes[1:]:
        axis.tick_params(which="both", labelleft=False)

    # As in plot_shifts: the slopes once the shared limits are known
    axes[0].set_xlim(*H_LIMITS)
    limits = axes[0].get_xlim(), axes[0].get_ylim()
    lowest = min(plot_slopes(axis, errors, SUBSCALE_SHIFT, domain, limits, SUBSCALE_DEGREES)
                 for axis, domain in zip(axes, DOMAINS))
    axes[0].set_xlim(limits[0])
    axes[0].set_ylim(lowest, limits[1][1])
    exponents = (-2.5, -2., -1.5)
    axes[0].set_xticks([10.**k for k in exponents], [f"$10^{{{k:g}}}$" for k in exponents])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    figure.savefig("out/gsbm_poisson_subscale.png", dpi=110, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def plot_domain_error(errors, domain_errors):
    """The convergence of DOMAIN_ERROR_SHIFT inside the domain on every
    domain, side by side on one y-axis, with its alpha-weighted error faded
    beneath and the slopes through the error inside the domain."""
    figure, axes = plt.subplots(1, len(DOMAINS), figsize=(6.5 * len(DOMAINS), 8.5), sharex=True, sharey=True,
                                layout="constrained")
    styles = [matplotlib.lines.Line2D([], [], color="black", marker="o", markersize=4.5, linewidth=2.2),
              matplotlib.lines.Line2D([], [], color="black", marker="o", markersize=3.5, linewidth=1.4, alpha=.5)]
    for axis, domain in zip(axes, DOMAINS):
        plot_convergence(axis, errors, DOMAIN_ERROR_SHIFT, domain, None, faded=True)
        plot_convergence(axis, domain_errors, DOMAIN_ERROR_SHIFT, domain, domain.capitalize())
        axis.get_legend().remove()
        axis.set_xlabel("$h$")
    # One legend below the panels, as the curves fill every corner of them
    handles, labels = max((axis.get_legend_handles_labels() for axis in axes), key=lambda pair: len(pair[1]))
    legend = figure.legend(handles + styles, labels + [r"in $\Omega$", r"$\alpha_h$-weighted, Eq. (48)"],
                           loc="upper center", bbox_to_anchor=(.5, 0.), ncols=len(labels) + len(styles))
    axes[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    for axis in axes[1:]:
        axis.tick_params(which="both", labelleft=False)

    # As in plot_shifts: the slopes once the shared limits are known
    axes[0].set_xlim(*H_LIMITS)
    limits = axes[0].get_xlim(), axes[0].get_ylim()
    lowest = min(plot_slopes(axis, domain_errors, DOMAIN_ERROR_SHIFT, domain, limits)
                 for axis, domain in zip(axes, DOMAINS))
    axes[0].set_xlim(limits[0])
    axes[0].set_ylim(lowest, limits[1][1])
    exponents = (-2.5, -2., -1.5)
    axes[0].set_xticks([10.**k for k in exponents], [f"$10^{{{k:g}}}$" for k in exponents])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    figure.savefig("out/gsbm_poisson_domain_error.png", dpi=110, bbox_inches="tight", pad_inches=.1,
                   bbox_extra_artists=[legend])
    plt.close(figure)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    errors = read_errors()
    for domain in DOMAINS:
        plot_degrees(domain)
        plot_shifts(domain, errors)
    plot_subscale(errors)
    domain_errors = read_errors("relative_error_domain")
    if any(shift == DOMAIN_ERROR_SHIFT for shift, _, _, _ in domain_errors):
        plot_domain_error(errors, domain_errors)
