"""
The figures of the benchmark, as those of the IGUANA study, from the outputs
of MainKratos.py and convergence.py in out/:

- out/gsbm_poisson_degrees_<domain>.png: u_h over the whole unit square, its
  extension outside the domain included, at the degrees 1 to 4 on N = 40, on
  one color scale, linear in [-1, 1] and logarithmic beyond, each titled with
  its degree;
- out/gsbm_poisson_shifts_<domain>.png: on the left, the field at p = 5 on
  the 40 x 40 grid of the json; on the right, a column for every shift in
  SHIFTS with its convergence at the degrees 1 to 5, relative errors against
  h from N = 40, with the slope h^(p+1) of every degree through its error on
  N = 80;
- out/gsbm_poisson_standard.png: the same convergence for the standard
  Taylor shift of the SBM papers, without the shifted subscale, a panel per
  domain;
- out/gsbm_poisson_penalties.png: that shift at p = 3 for every combination
  of the Nitsche and ghost penalty factors of penalties.py, against the shift
  with the subscale, a panel per domain.

The Taylor shift has its shifted subscale on (USE_RESIDUAL_TERMS), off in
config/MaterialParameters_standard_p<p>.json, and the ghost penalty of Sticko
and Kreiss, kappa_p = 0.5 / ((p!)^2 (2p + 1)), of
config/MaterialParameters_p<p>.json:

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

DOMAINS = ("circle", "flower", "parallelogram", "heart")
DEGREES = (1, 2, 3, 4)
COLOURS = {1: "C0", 2: "C1", 3: "C2", 4: "C3", 5: "C4"}

# The degrees of the convergence plots
CONVERGENCE_DEGREES = (1, 2, 3, 4, 5)

# The shift of the fields, the mesh of the fields at all degrees, and the
# degree and mesh of the field beside the convergence
FIELD_SHIFT = "taylor"
DEGREES_CELLS = 40
FIELD = (5, 40)

# The convergence plots, a column per shift: from N = 40, with the slopes
# through N = 80
MIN_CELLS = 40
ANCHOR_CELLS = 80
SHIFTS = (("taylor", "Taylor shift, shifted subscale on"),)

# The standard Taylor shift of the SBM papers, without the shifted subscale,
# in its own figure
STANDARD_SHIFT = "standard"

# The degree of the penalty study of the standard shift, out/penalties_p3.csv,
# and the degrees of the shift with the subscale beside it, with their colour
# and marker
PENALTY_DEGREE = 3
PENALTY_REFERENCES = ((3, "C3", "o"), (2, "C1", "s"))

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


def read_errors():
    """Relative errors in out/convergence_p<p>.csv, keyed by shift, domain,
    degree and mesh."""
    errors = {}
    for path in Path("out").glob("convergence_p*.csv"):
        degree = int(path.stem.removeprefix("convergence_p"))
        with open(path) as f:
            for row in csv.DictReader(f):
                domain, shift = row["case"].split("_", 1)
                errors[(shift, domain, degree, int(row["N"]))] = float(row["relative_error"])
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
    figure, axes = plt.subplots(1, len(DEGREES), figsize=(24, 6.8), layout="constrained")

    for axis, field, degree in zip(axes, fields, DEGREES):
        image = draw_field(axis, field, skin, DEGREES_CELLS, bound, f"$p = {degree}$")

    field_colorbar(figure, image, axes, bound)
    figure.savefig(f"out/gsbm_poisson_degrees_{domain}.png", dpi=100, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def plot_convergence(axis, errors, shift, domain, title):
    for degree in CONVERGENCE_DEGREES:
        pairs = sorted((cells, error) for (s, d, p, cells), error in errors.items()
                       if (s, d, p) == (shift, domain, degree) and cells >= MIN_CELLS)
        if len(pairs) < 2:
            continue
        cells, values = map(np.array, zip(*pairs))
        axis.loglog(1. / cells, values, color=COLOURS[degree], marker="o", markersize=7, linewidth=2.2,
                    label=f"$p = {degree}$")
    if title:
        axis.set_title(title)
    axis.grid(True, which="both", alpha=.3)
    axis.legend(loc="lower right")


def plot_slopes(axis, errors, shift, domain, limits):
    """The optimal slope of each degree through its error on N = 80, across
    the axis, with a small triangle at the finest mesh of the degree, moved
    along the slope to coarser meshes while it overlaps the triangle of a
    lower degree; returns the lowest value the triangles reach."""
    lowest = limits[1][0]
    placed = []
    for degree in CONVERGENCE_DEGREES:
        key = (shift, domain, degree, ANCHOR_CELLS)
        if key not in errors:
            continue

        def reference(h, degree=degree, anchor=errors[key]):
            return .6 * anchor * (h * ANCHOR_CELLS)**(degree + 1)

        h = np.array(limits[0])
        axis.plot(h, reference(h), color="black", linewidth=1.3)
        finest = max(cells for (s, d, p, cells) in errors if (s, d, p) == (shift, domain, degree))
        low, high = 1. / finest, 1.25 / finest

        # With some room for the labels, below and to the right
        def overlaps(low, high):
            return any(low < right * 1.1 and left < high * 1.1
                       and reference(low) / 1.5 < top and bottom / 1.5 < reference(high)
                       for left, right, bottom, top in placed)

        while overlaps(low, high):
            low, high = 1.25 * low, 1.25 * high
        bottom, top = reference(low), reference(high)
        placed.append((low, high, bottom, top))
        lowest = min(lowest, bottom / 2.)
        axis.plot([low, high, high, low], [bottom, bottom, top, bottom], color="black", linewidth=1.3)
        axis.text(np.sqrt(low * high), bottom / 1.1, "1", fontsize=15, ha="center", va="top")
        axis.text(high * 1.03, np.sqrt(bottom * top), f"{degree + 1}", fontsize=15, ha="left", va="center")
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

    # Column titles only to tell several shifts apart
    reference = read_reference()
    for axis, (shift, title) in zip(axes, SHIFTS):
        plot_convergence(axis, errors, shift, domain, title if len(SHIFTS) > 1 else None)
        plot_reference(axis, reference, domain)
        axis.set_xlabel("$h$")
    first_axis.set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    for axis in axes[1:]:
        axis.tick_params(which="both", labelleft=False)

    # The slopes once the shared limits of all panels are known, h from just
    # below 1/320 to the half decade above 1/40
    first_axis.set_xlim(10.**-2.55, 10.**-1.5)
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


def plot_standard_shift(errors):
    """The convergence of the standard Taylor shift, without the shifted
    subscale, on every domain, side by side on one y-axis, each titled with
    its domain."""
    figure, axes = plt.subplots(1, len(DOMAINS), figsize=(6.5 * len(DOMAINS), 8.5), sharex=True, sharey=True,
                                layout="constrained")
    reference = read_reference()
    for axis, domain in zip(axes, DOMAINS):
        plot_convergence(axis, errors, STANDARD_SHIFT, domain, domain.capitalize())
        plot_reference(axis, reference, domain)
        axis.set_xlabel("$h$")
    axes[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    for axis in axes[1:]:
        axis.tick_params(which="both", labelleft=False)

    # As in plot_shifts: the slopes once the shared limits are known
    axes[0].set_xlim(10.**-2.55, 10.**-1.5)
    limits = axes[0].get_xlim(), axes[0].get_ylim()
    lowest = min(plot_slopes(axis, errors, STANDARD_SHIFT, domain, limits) for axis, domain in zip(axes, DOMAINS))
    axes[0].set_xlim(limits[0])
    axes[0].set_ylim(lowest, limits[1][1])
    exponents = (-2.5, -2., -1.5)
    axes[0].set_xticks([10.**k for k in exponents], [f"$10^{{{k:g}}}$" for k in exponents])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    figure.savefig("out/gsbm_poisson_standard.png", dpi=110, bbox_inches="tight", pad_inches=.1)
    plt.close(figure)


def read_penalties():
    """Relative errors in out/penalties_p3.csv, keyed by domain, the factors
    chi and kappa, and mesh."""
    errors = {}
    path = Path("out/penalties_p3.csv")
    if path.exists():
        with open(path) as f:
            for row in csv.DictReader(f):
                key = (row["case"].split("_", 1)[0], float(row["chi"]), float(row["kappa"]), int(row["N"]))
                errors[key] = float(row["relative_error"])
    return errors


def plot_penalties(errors, penalties):
    """The convergence of the standard Taylor shift at p = 3 for every
    combination of the penalty factors, a colour per chi and a line style per
    kappa, against the shift with the subscale at the factors of the
    materials, at p = 3 in red and p = 2 in orange, a panel per domain, with
    the slope h^4 through the subscale at p = 3 on N = 80."""
    chis = sorted({chi for _, chi, _, _ in penalties})
    kappas = sorted({kappa for _, _, kappa, _ in penalties})
    colours = dict(zip(chis, plt.cm.viridis(np.linspace(0., .85, len(chis)))))
    styles = dict(zip(kappas, (":", "-.", "--", "-")))
    subscale = SHIFTS[0][0]
    figure, axes = plt.subplots(1, len(DOMAINS), figsize=(6.5 * len(DOMAINS), 8.5), sharex=True, sharey=True,
                                layout="constrained")
    for axis, domain in zip(axes, DOMAINS):
        for chi in chis:
            for kappa in kappas:
                pairs = sorted((cells, error) for (d, c, k, cells), error in penalties.items()
                               if (d, c, k) == (domain, chi, kappa))
                if len(pairs) > 1:
                    cells, values = map(np.array, zip(*pairs))
                    axis.loglog(1. / cells, values, color=colours[chi], linestyle=styles[kappa], linewidth=1.8,
                                marker="o", markersize=4)
        for degree, colour, marker in PENALTY_REFERENCES:
            pairs = sorted((cells, error) for (s, d, p, cells), error in errors.items()
                           if (s, d, p) == (subscale, domain, degree) and cells >= MIN_CELLS)
            cells, values = map(np.array, zip(*pairs))
            axis.loglog(1. / cells, values, color=colour, linewidth=3.6, marker=marker, markersize=9, zorder=5)
        anchor = errors[(subscale, domain, PENALTY_DEGREE, ANCHOR_CELLS)]
        h = np.array([10.**-2.55, 10.**-1.5])
        axis.loglog(h, .3 * anchor * (h * ANCHOR_CELLS)**(PENALTY_DEGREE + 1), color="black", linewidth=1.,
                    linestyle=(0, (1, 3)))
        axis.set_title(domain.capitalize())
        axis.set_xlabel("$h$")
        axis.grid(True, which="both", alpha=.3)
    axes[0].set_ylabel(r"$\Vert u - u_h\Vert \,/\, \Vert u\Vert$")
    for axis in axes[1:]:
        axis.tick_params(which="both", labelleft=False)
    axes[0].set_xlim(10.**-2.55, 10.**-1.5)
    exponents = (-2.5, -2., -1.5)
    axes[0].set_xticks([10.**k for k in exponents], [f"$10^{{{k:g}}}$" for k in exponents])
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # Below the panels, a row of the factors and one of the references
    line = matplotlib.lines.Line2D
    factors = ([line([], [], color=colours[chi], linewidth=2.4) for chi in chis]
               + [line([], [], color="0.3", linestyle=styles[kappa], linewidth=1.8) for kappa in kappas])
    factor_labels = [f"$\\chi = {chi:g}$" for chi in chis] + [f"$\\kappa = {kappa:g}$" for kappa in kappas]
    references = ([line([], [], color=colour, linewidth=3.6, marker=marker, markersize=9)
                   for _, colour, marker in PENALTY_REFERENCES]
                  + [line([], [], color="black", linewidth=1., linestyle=(0, (1, 3)))])
    reference_labels = [f"$p = {degree}$" for degree, _, _ in PENALTY_REFERENCES] + [f"$h^{PENALTY_DEGREE + 1}$"]
    legends = [figure.legend(factors, factor_labels, loc="upper center", bbox_to_anchor=(.5, 0.),
                             ncols=len(factors), title="without the subscale, $p = 3$"),
               figure.legend(references, reference_labels, loc="upper center", bbox_to_anchor=(.5, -.11),
                             ncols=len(references), handlelength=4.,
                             title="with the subscale, $\\chi = 20$, $\\kappa = 0.5$")]
    figure.savefig("out/gsbm_poisson_penalties.png", dpi=110, bbox_inches="tight", pad_inches=.1,
                   bbox_extra_artists=legends)
    plt.close(figure)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    errors = read_errors()
    for domain in DOMAINS:
        plot_degrees(domain)
        plot_shifts(domain, errors)
    plot_standard_shift(errors)
    penalties = read_penalties()
    if penalties:
        plot_penalties(errors, penalties)
