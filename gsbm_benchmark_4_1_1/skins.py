"""
Writes the skins of the three domains of Section 4.1.1 of Colomés, Modderman
and Scovazzi, CMAME 452 (2026) 118748, Eqs. (44)-(46), and of a heart of our
own, to config/, as skin_<domain>_nurbs.json: counterclockwise NURBS curves
for ImportNurbsSbmModeler, the circle exactly as one curve, the flower as one
quintic spline interpolating it, the parallelogram exactly as one line per
side, and the heart exactly as two circular lobes, rational quadratic arcs
meeting at a re-entrant corner at the top (the notch), and two lines tangent
to them meeting at a convex corner at the bottom (the tip):

    tip -> right lobe -> notch -> left lobe -> tip

    python3 skins.py
"""

import json
from pathlib import Path

import numpy as np
from scipy.interpolate import make_interp_spline

FLOWER_SAMPLES = 512
PARALLELOGRAM = ((0.21, 0.21), (0.64, 0.31), (0.83, 0.83), (0.37, 0.73))  # v1 to v4
LOBE_RADIUS = 0.17
LOBE_CENTERS = (np.array([0.63, 0.6]), np.array([0.37, 0.6]))  # right, left
TIP = np.array([0.5, 0.18])
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


def arc(center, radius, start, end):
    """The arc of a circle from the angle start to end, counterclockwise, as
    one NURBS curve of rational quadratic segments of at most 90 degrees."""
    sweep = end - start
    segments = int(np.ceil(sweep / (np.pi / 2.) - 1e-12))
    step = sweep / segments
    weight = np.cos(step / 2.)
    points, weights = [center + radius * np.array([np.cos(start), np.sin(start)])], [1.]
    for i in range(segments):
        middle, last = start + (i + .5) * step, start + (i + 1) * step
        # The middle control point, where the tangents at both ends meet
        points.append(center + radius / weight * np.array([np.cos(middle), np.sin(middle)]))
        points.append(center + radius * np.array([np.cos(last), np.sin(last)]))
        weights += [weight, 1.]
    inner = [k for i in range(1, segments) for k in (i / segments,) * 2]
    return 2, [0., 0., 0., *inner, 1., 1., 1.], np.array(points), weights


def line(first, last):
    """A line as a NURBS curve of degree 1."""
    return 1, [0., 0., 1., 1.], np.array([first, last]), [1., 1.]


def tangent_angle(center, radius, point, turn):
    """The angle on a circle of the point where a line from an outside point
    touches it: the angle of the point seen from the center, turned
    counterclockwise (turn = 1) or clockwise (turn = -1)."""
    to_point = point - center
    return np.arctan2(to_point[1], to_point[0]) + turn * np.arccos(radius / np.linalg.norm(to_point))


def heart_angles():
    """The angles on the right and left lobes of the notch, where they meet,
    and of the points where the lines from the tip touch them."""
    right, left = LOBE_CENTERS
    half_distance = (right[0] - left[0]) / 2.
    notch = np.array([(right[0] + left[0]) / 2., right[1] + np.sqrt(LOBE_RADIUS**2 - half_distance**2)])
    # The lines touch the lobes on their outer sides: the tip is below, so
    # counterclockwise from it on the right lobe and clockwise on the left one
    return {"notch_right": np.arctan2(*(notch - right)[::-1]), "notch_left": np.arctan2(*(notch - left)[::-1]),
            "touch_right": tangent_angle(right, LOBE_RADIUS, TIP, 1.),
            "touch_left": tangent_angle(left, LOBE_RADIUS, TIP, -1.) % (2. * np.pi)}


def on_lobe(center, angle):
    """The point of a lobe at an angle."""
    return center + LOBE_RADIUS * np.array([np.cos(angle), np.sin(angle)])


def heart_nurbs():
    """The four curves of the heart, counterclockwise from the tip."""
    right, left = LOBE_CENTERS
    angles = heart_angles()
    return [line(TIP, on_lobe(right, angles["touch_right"])),
            arc(right, LOBE_RADIUS, angles["touch_right"], angles["notch_right"]),
            arc(left, LOBE_RADIUS, angles["notch_left"], angles["touch_left"]),
            line(on_lobe(left, angles["touch_left"]), TIP)]


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
    write_nurbs(config / "skin_heart_nurbs.json", heart_nurbs())
