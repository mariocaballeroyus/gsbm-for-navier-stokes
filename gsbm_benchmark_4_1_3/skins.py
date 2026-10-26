"""
Writes the skins of Section 4.1.3 of Colomés, Modderman and Scovazzi, CMAME
452 (2026) 118748, to config/skins/, one per position d of the moving circle,
as skin_d<d>_nurbs.json: counterclockwise NURBS curves for
ImportNurbsSbmModeler, all exact. Two circles of radius 0.15, one fixed around
(0.2, 0.5) and the other around (0.5 + d, 0.5), d from -0.2 to 0.2 in steps of
0.0125:

- d >= 0: the two circles, apart or, at d = 0, touching at (0.35, 0.5);
- d < 0: the boundary of their union, the arc of each circle outside the
  other, meeting at the two points where they intersect.

The skins of other positions go to out/skins/, written by ensure_skin when
first needed.

    python3 skins.py
"""

import json
from pathlib import Path

import numpy as np

RADIUS = 0.15
FIXED_CENTER = np.array([0.2, 0.5])
D_VALUES = tuple(k * 0.0125 for k in range(-16, 17))
LAYER = "outer"


def moving_center(d):
    return np.array([0.5 + d, 0.5])


def position_name(d):
    """A position as in file names: signed, with 4 decimals, or 5 if it needs
    them."""
    d = round(d, 5) + 0.  # + 0. turns -0 into 0
    return f"{d:+.4f}" if round(d, 4) == d else f"{d:+.5f}"


def skin_file(d):
    """The skin of a position, relative to this folder: in config/skins/ for
    those of D_VALUES, in out/skins/ for others."""
    folder = "config" if any(abs(d - value) < 1e-9 for value in D_VALUES) else "out"
    return f"{folder}/skins/skin_d{position_name(d)}_nurbs.json"


def ensure_skin(d):
    """The skin of a position, written if missing."""
    path = Path(__file__).parent / skin_file(d)
    if not path.exists():
        write_nurbs(path, circles_nurbs(d))
    return skin_file(d)


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


def intersection_angle(d):
    """The angle, from the line of the centers, at which the fixed circle
    meets the moving one, for d < 0: the moving circle meets it at pi minus
    that angle, as their radii are equal."""
    return np.arccos((moving_center(d)[0] - FIXED_CENTER[0]) / (2. * RADIUS))


def circles_nurbs(d):
    """The curves of the skin at a position d."""
    moving = moving_center(d)
    if d >= 0.:
        # Starting at their far sides, so that the point where they touch at
        # d = 0 is inside both curves rather than an endpoint shared by them
        return [arc(FIXED_CENTER, RADIUS, np.pi, 3. * np.pi), arc(moving, RADIUS, 0., 2. * np.pi)]
    # From the upper intersection around the fixed circle to the lower one,
    # and back around the moving circle
    angle = intersection_angle(d)
    return [arc(FIXED_CENTER, RADIUS, angle, 2. * np.pi - angle), arc(moving, RADIUS, angle - np.pi, np.pi - angle)]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"Lines": lines}, indent=4) + "\n")


if __name__ == "__main__":
    here = Path(__file__).parent
    for d in D_VALUES:
        write_nurbs(here / skin_file(d), circles_nurbs(d))
