"""
Processes of the GSBM Poisson benchmark, chosen by "process_name":

- BoundaryDataProcess: g and, for the stabilized shift, the derivatives of f
  at the closest skin points;
- L2ErrorProcess: the alpha-weighted L2 error of Eq. (48);
- VtkFieldsOutputProcess: the fields on a uniform grid of the background, for
  ParaView.

Like VolumeFractionGsbmProcess, they assume uv = xy on the background patch.
"""

from pathlib import Path

import numpy as np
import sympy as sp
from matplotlib.path import Path as Polygon
from scipy.interpolate import BSpline

import KratosMultiphysics as KM
import KratosMultiphysics.IgaApplication as IGA


def Factory(settings, model):
    if not isinstance(settings, KM.Parameters):
        raise Exception("Expected input shall be a Parameters object, encapsulating a json string")
    processes = {
        "BoundaryDataProcess": BoundaryDataProcess,
        "L2ErrorProcess": L2ErrorProcess,
        "VtkFieldsOutputProcess": VtkFieldsOutputProcess,
    }
    return processes[settings["process_name"].GetString()](model, settings["Parameters"])


def parse_function(expression):
    """f(x, y, t) from an expression of the Kratos function parser."""
    function = KM.GenericFunctionUtility(expression)
    return lambda x, y, t=0.0: function.CallFunction(x, y, 0.0, t, x, y, 0.0)


def span_indices(spans, values):
    """Knot span of every value: a value on a knot goes to the span on its
    right, up to round-off, and the last span includes its end."""
    tolerance = 1e-10 * (spans[-1] - spans[0])
    return np.clip(np.searchsorted(spans, values + tolerance, side="right") - 1, 0, len(spans) - 2)


def knot_span_volume_fractions(elements, spans):
    """Volume fraction of every knot span of the background, from its elements."""
    data = np.array([(point.X, point.Y, element.GetValue(IGA.VOLUME_FRACTION))
                     for element in elements for point in (element.GetGeometry().Center(),)])
    alpha = np.zeros((len(spans[0]) - 1, len(spans[1]) - 1))
    alpha[span_indices(spans[0], data[:, 0]), span_indices(spans[1], data[:, 1])] = data[:, 2]
    return alpha


def background_values(background, variable, grids):
    """A nodal variable of the background patch on the tensor grid
    grids[0] x grids[1], indexed [x, y]."""
    bases = []
    for grid, knots, degree in ((grids[0], background.KnotsU(), background.PolynomialDegreeU()),
                                (grids[1], background.KnotsV(), background.PolynomialDegreeV())):
        # Kratos stores the end knots once less than a full knot vector
        knots = np.array(knots)
        knots = np.concatenate([[knots[0]], knots, [knots[-1]]])
        bases.append(BSpline.design_matrix(grid, knots, degree))
    # Control point values, ordered u fastest
    coefficients = np.array([node.GetSolutionStepValue(variable) for node in background])
    coefficients = coefficients.reshape(background.NumberOfControlPointsV(), -1).T
    return (bases[1] @ (bases[0] @ coefficients).T).T


class BoundaryDataProcess(KM.Process):
    """Assigns at the closest skin points what the GSBM Dirichlet conditions
    read there: g, to the unknown, and for the stabilized shift
    VOLUME_SOURCE_DERIVATIVES, f and its Cartesian derivatives of orders 0 to
    p - 2, every order n as d_x^(n-k) d_y^k, k = 0 to n. g and f are
    expressions of the Kratos function parser, f differentiated with SymPy.
    The nodes are found when the step starts, as VolumeFractionGsbmProcess
    creates them."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "background_model_part_name": "",
            "geometry_name": "",
            "variable_name": "",
            "boundary_value": "",
            "volume_source": ""
        }"""))
        self.model = model
        self.settings = settings
        self.variable = KM.KratosGlobals.GetVariable(settings["variable_name"].GetString())
        self.symbols = sp.symbols("x y z t")
        locals_ = {str(symbol): symbol for symbol in self.symbols}
        self.g, self.f = (sp.sympify(settings[name].GetString().replace("^", "**"), locals=locals_)
                          for name in ("boundary_value", "volume_source"))

    def ExecuteInitializeSolutionStep(self):
        background_model_part = self.model[self.settings["background_model_part_name"].GetString()]
        background = background_model_part.GetGeometry(self.settings["geometry_name"].GetString())
        degree = max(background.PolynomialDegreeU(), background.PolynomialDegreeV())
        time = background_model_part.ProcessInfo[KM.TIME]
        nodes = list(self.model[self.settings["model_part_name"].GetString()].Nodes)
        if not nodes:
            return  # no cut facets
        x, y = np.array([[node.X, node.Y] for node in nodes]).T

        x_, y_, z_, t_ = self.symbols

        def derivatives(expression, orders):
            """Values at the nodes of every d_x^(n-k) d_y^k expression, n in orders."""
            columns = []
            for n in orders:
                for k in range(n + 1):
                    derivative = expression.diff(*[x_] * (n - k), *[y_] * k) if n > 0 else expression
                    values = sp.lambdify(self.symbols, derivative, "numpy")(x, y, 0., time)
                    columns.append(np.broadcast_to(values, x.shape))
            return np.array(columns).T.reshape(len(nodes), -1)

        g = derivatives(self.g, [0])[:, 0]
        f_derivatives = derivatives(self.f, range(degree - 1))
        for node, value, f_row in zip(nodes, g, f_derivatives):
            node.SetValue(self.variable, value)
            node.SetValue(IGA.VOLUME_SOURCE_DERIVATIVES, KM.Vector(f_row.tolist()))


class L2ErrorProcess(KM.Process):
    """Alpha-weighted L2 norm of the error, Eq. (48): the sum over the knot
    spans of alpha times the integral of (u - u_h)^2, with
    number_of_integration_points_per_span Gauss points per direction (0 for
    p + 1, the Gauss points of the elements); and the same norm of u, for the
    relative error."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "geometry_name": "",
            "elements_model_part_name": "",
            "variable_name": "",
            "exact_solution": "",
            "number_of_integration_points_per_span": 0
        }"""))
        self.model = model
        self.settings = settings
        self.variable = KM.KratosGlobals.GetVariable(settings["variable_name"].GetString())
        self.exact = parse_function(settings["exact_solution"].GetString())
        self.l2_error = None
        self.l2_norm = None

    def ExecuteFinalizeSolutionStep(self):
        model_part = self.model[self.settings["model_part_name"].GetString()]
        background = model_part.GetGeometry(self.settings["geometry_name"].GetString())
        spans = [np.array(background.SpansLocalSpace(direction)) for direction in range(2)]
        alpha = knot_span_volume_fractions(
            self.model[self.settings["elements_model_part_name"].GetString()].Elements, spans)

        # Gauss points and weights of all knot spans, per direction
        points_per_span, grids, weights = [], [], []
        for direction, degree in enumerate((background.PolynomialDegreeU(), background.PolynomialDegreeV())):
            n = self.settings["number_of_integration_points_per_span"].GetInt() or degree + 1
            points, point_weights = np.polynomial.legendre.leggauss(n)
            lower, size = spans[direction][:-1, None], np.diff(spans[direction])[:, None]
            points_per_span.append(n)
            grids.append((lower + size * (points + 1.) / 2.).ravel())
            weights.append((size * point_weights / 2.).ravel())

        time = model_part.ProcessInfo[KM.TIME]
        u_h = background_values(background, self.variable, grids)
        X, Y = np.meshgrid(*grids, indexing="ij")
        u = np.vectorize(lambda x, y: self.exact(x, y, time))(X, Y)
        def norm(values):
            """Alpha-weighted L2 norm, from the integral over every knot span."""
            squared = (values**2 * np.outer(*weights)).reshape(
                alpha.shape[0], points_per_span[0], alpha.shape[1], points_per_span[1]).sum(axis=(1, 3))
            return np.sqrt(np.sum(alpha * squared))

        self.l2_error = norm(u - u_h)
        self.l2_norm = norm(u)
        KM.Logger.PrintInfo("L2ErrorProcess", f"L2 error: {self.l2_error:.6e}, "
                            f"relative: {self.l2_error / self.l2_norm:.6e}")


class VtkFieldsOutputProcess(KM.OutputProcess):
    """Writes u_h, u_exact, error, volume_fraction and inside on a uniform
    grid of the background (u_h also outside the domain) as legacy VTK
    structured points, <output_file_name>.vtk, and the skin polygon,
    <skin_output_file_name>.vtk."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "geometry_name": "",
            "elements_model_part_name": "",
            "skin_model_part_name": "",
            "variable_name": "",
            "exact_solution": "",
            "number_of_samples": 401,
            "output_file_name": "",
            "skin_output_file_name": ""
        }"""))
        self.model = model
        self.settings = settings
        self.variable = KM.KratosGlobals.GetVariable(settings["variable_name"].GetString())
        self.exact = parse_function(settings["exact_solution"].GetString())

    def IsOutputStep(self):
        return True

    def PrintOutput(self):
        model_part = self.model[self.settings["model_part_name"].GetString()]
        background = model_part.GetGeometry(self.settings["geometry_name"].GetString())
        time = model_part.ProcessInfo[KM.TIME]
        samples = self.settings["number_of_samples"].GetInt()
        spans = [np.array(background.SpansLocalSpace(direction)) for direction in range(2)]
        grids = [np.linspace(spans[direction][0], spans[direction][-1], samples) for direction in range(2)]

        u_h = background_values(background, self.variable, grids)
        X, Y = np.meshgrid(*grids, indexing="ij")
        u = np.vectorize(lambda x, y: self.exact(x, y, time))(X, Y)

        skin = [condition.GetGeometry()[0] for condition in
                self.model[self.settings["skin_model_part_name"].GetString()].Conditions]
        skin = np.array([[node.X, node.Y] for node in skin])
        inside = Polygon(skin).contains_points(np.column_stack([X.ravel(), Y.ravel()]))
        inside = inside.reshape(X.shape)

        alpha = knot_span_volume_fractions(
            self.model[self.settings["elements_model_part_name"].GetString()].Elements, spans)
        alpha = alpha[np.ix_(span_indices(spans[0], grids[0]), span_indices(spans[1], grids[1]))]

        path = Path(self.settings["output_file_name"].GetString())
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.with_suffix(".vtk"), "w") as f:
            f.write("# vtk DataFile Version 3.0\nGSBM Poisson\nASCII\nDATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {samples} {samples} 1\nORIGIN {grids[0][0]} {grids[1][0]} 0\n")
            f.write(f"SPACING {grids[0][1] - grids[0][0]} {grids[1][1] - grids[1][0]} 1\n")
            f.write(f"POINT_DATA {samples * samples}\n")
            # VTK orders the points with x fastest, so the [x, y] arrays go transposed
            for name, field in (("u_h", u_h), ("u_exact", u), ("error", u_h - u),
                                ("volume_fraction", alpha), ("inside", inside * 1.)):
                f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, field.T.reshape(-1), fmt="%.10e")

        skin_path = Path(self.settings["skin_output_file_name"].GetString()).with_suffix(".vtk")
        skin_path.parent.mkdir(parents=True, exist_ok=True)
        with open(skin_path, "w") as f:
            f.write("# vtk DataFile Version 3.0\nGSBM skin\nASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {len(skin)} double\n")
            np.savetxt(f, np.column_stack([skin, np.zeros(len(skin))]), fmt="%.10e")
            f.write(f"LINES 1 {len(skin) + 2}\n{len(skin) + 1} "
                    + " ".join(map(str, range(len(skin)))) + " 0\n")

        KM.Logger.PrintInfo("VtkFieldsOutputProcess",
                            f"{path.name}: u_h in [{u_h.min():.2f}, {u_h.max():.2f}], max |u - u_h| "
                            f"in the domain {np.abs(u_h - u)[inside].max():.2e}")
