"""
Writes the skins of the three domains of Section 4.1.1 of Colomés, Modderman
and Scovazzi, CMAME 452 (2026) 118748, Eqs. (44)-(46), to config/, as
skin_<domain>_nurbs.json: counterclockwise NURBS curves for
ImportNurbsSbmModeler, the circle exactly as one curve, the flower as one
quintic spline interpolating it, and the parallelogram exactly as one line
per side.

    python3 skins.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

FLOWER_SAMPLES = 512
PARALLELOGRAM = ((0.21, 0.21), (0.64, 0.31), (0.83, 0.83), (0.37, 0.73))  # v1 to v4
LAYER = "outer"


def circle_nurbs():
    """Eq. (44) exactly: four rational quadratic quarter arcs, 9 control points."""
    corners = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0)]
    control_points = 0.5 + 0.3 * np.array(corners, dtype=float)
    weights = [1., np.sqrt(.5)] * 4 + [1.]
    knots = [0., 0., 0., .25, .25, .5, .5, .75, .75, 1., 1., 1.]
    return 2, knots, control_points, weights


def flower_point(t, derivative=0):
    """Eq. (45), radius R(t) = 0.3 (1 + 0.3 sin(5t)) around (0.5, 0.5), or its
    derivatives in t."""
    r = (0.3 * (1. + 0.3 * np.sin(5. * t)), 0.45 * np.cos(5. * t), -2.25 * np.sin(5. * t))
    e = np.column_stack([np.cos(t), np.sin(t)])
    f = np.column_stack([-np.sin(t), np.cos(t)])
    if derivative == 0:
        return 0.5 + r[0][:, None] * e
    if derivative == 1:
        return r[1][:, None] * e + r[0][:, None] * f
    return (r[2] - r[0])[:, None] * e + 2. * r[1][:, None] * f


def flower_nurbs():
    """Eq. (45) as a quintic spline interpolating FLOWER_SAMPLES + 1 points,
    with its first and second derivatives where it closes, on knots in
    [0, 1]."""
    t = np.linspace(0., 2. * np.pi, FLOWER_SAMPLES + 1)
    ends = [(1, flower_point(np.zeros(1), 1)[0]), (2, flower_point(np.zeros(1), 2)[0])]
    spline = make_interp_spline(t, flower_point(t), k=5, bc_type=(ends, ends))
    return 5, spline.t / (2. * np.pi), spline.c, np.ones(len(spline.c))


def parallelogram_nurbs():
    """Eq. (46) exactly: the sides from v1 to v4, each a line, a NURBS curve
    of degree 1."""
    vertices = np.array(PARALLELOGRAM)
    return [(1, [0., 0., 1., 1.], vertices[[i, (i + 1) % 4]], [1., 1.]) for i in range(4)]


def write_nurbs(path, curves):
    """NURBS curves, each (degree, knots, control points, weights), in the
    layer LAYER, as ImportNurbsSbmModeler reads them."""
    lines = [{
        "Layer": LAYER,
        "id": i + 1,
        "pDegree": degree,
        "knotVector": list(map(float, knots)),
        "Weights": list(map(float, weights)),
        "CPCoordinates": [[float(x), float(y), 0.0] for x, y in control_points],
    } for i, (degree, knots, control_points, weights) in enumerate(curves)]
    path.write_text(json.dumps({"Lines": lines}, indent=4) + "\n")


if __name__ == "__main__":
    config = Path(__file__).parent / "config"
    write_nurbs(config / "skin_circle_nurbs.json", [circle_nurbs()])
    write_nurbs(config / "skin_flower_nurbs.json", [flower_nurbs()])
    write_nurbs(config / "skin_parallelogram_nurbs.json", parallelogram_nurbs())
