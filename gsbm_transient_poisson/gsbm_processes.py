"""
Processes of the GSBM transient Poisson benchmark on a moving domain, chosen by
"process_name":

- MoveSkinProcess: the skin moved rigidly by a displacement of time, for
  VolumeFractionGsbmProcess with update_each_step to find it there;
- ManufacturedSourceProcess: f = rho c du/dt - k lap u of a manufactured
  solution, or -k lap u alone at the step without BDF_COEFFICIENTS (the
  elliptic projection of the initial condition);
- BoundaryDataProcess: g at the closest skin points, and f and its
  derivatives at the cut facets, for the shifted subscale of the Taylor shift;
- ActiveSpansProcess: the problem on the knot spans with alpha > 0 alone;
- L2ErrorProcess: the alpha-weighted L2 error of Eq. (48), and the L2 error
  in the domain, at every step and over the time steps, Eqs. (54) and (55);
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
        "MoveSkinProcess": MoveSkinProcess,
        "ManufacturedSourceProcess": ManufacturedSourceProcess,
        "BoundaryDataProcess": BoundaryDataProcess,
        "ActiveSpansProcess": ActiveSpansProcess,
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


def background_point_values(background, variable, points, chunk=200000):
    """A nodal variable of the background patch at scattered points, rows
    (x, y), from the p + 1 functions of each direction that are nonzero there."""
    knots, degrees = [], (background.PolynomialDegreeU(), background.PolynomialDegreeV())
    for direction_knots in (background.KnotsU(), background.KnotsV()):
        direction_knots = np.array(direction_knots)
        knots.append(np.concatenate([[direction_knots[0]], direction_knots, [direction_knots[-1]]]))
    coefficients = np.array([node.GetSolutionStepValue(variable) for node in background])
    coefficients = coefficients.reshape(background.NumberOfControlPointsV(), -1).T
    values = np.empty(len(points))
    for start in range(0, len(points), chunk):
        local = []
        for direction in range(2):
            matrix = BSpline.design_matrix(points[start:start + chunk, direction], knots[direction],
                                           degrees[direction])
            local.append((matrix.indices.reshape(-1, degrees[direction] + 1)[:, 0],
                          matrix.data.reshape(-1, degrees[direction] + 1)))
        (first_u, basis_u), (first_v, basis_v) = local
        block = coefficients[first_u[:, None, None] + np.arange(degrees[0] + 1)[None, :, None],
                             first_v[:, None, None] + np.arange(degrees[1] + 1)[None, None, :]]
        values[start:start + chunk] = np.einsum("ka,kb,kab->k", basis_u, basis_v, block)
    return values


def domain_quadrature(polygon, cells, n, depth):
    """Points and weights of a quadrature over the part inside a polygon of
    square cells, rows (x0, y0, size), all cut by it: every cell the polygon
    cuts is split into four, depth times, with n x n Gauss points in every
    leaf and, in the leaves still cut, only the points inside. A cell is cut
    when its corners and center are not all on one side."""
    gauss, gauss_weights = np.polynomial.legendre.leggauss(n)
    offsets = np.stack(np.meshgrid((gauss + 1.) / 2., (gauss + 1.) / 2., indexing="ij"), -1).reshape(-1, 2)
    unit_weights = np.outer(gauss_weights, gauss_weights).ravel() / 4.
    probes = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.], [.5, .5]])
    children = np.array([[0., 0.], [.5, 0.], [0., .5], [.5, .5]])
    points, weights = [], []

    def leaves(cells, keep_inside_only):
        cell_points = (cells[:, None, :2] + cells[:, None, 2:] * offsets[None]).reshape(-1, 2)
        cell_weights = (cells[:, 2:]**2 * unit_weights[None]).ravel()
        if keep_inside_only:
            inside = polygon.contains_points(cell_points)
            cell_points, cell_weights = cell_points[inside], cell_weights[inside]
        points.append(cell_points)
        weights.append(cell_weights)

    for level in range(depth + 1):
        if level > 0:
            inside = polygon.contains_points(
                (cells[:, None, :2] + cells[:, None, 2:] * probes[None]).reshape(-1, 2)).reshape(-1, len(probes))
            leaves(cells[inside.all(axis=1)], False)
            cells = cells[inside.any(axis=1) & ~inside.all(axis=1)]
        if level == depth:
            leaves(cells, True)
        else:
            # The four children of every cell, of half its size
            corners = (cells[:, None, :2] + cells[:, None, 2:] * children[None]).reshape(-1, 2)
            cells = np.column_stack([corners, np.repeat(cells[:, 2] / 2., len(children))])
    return np.concatenate(points), np.concatenate(weights)


class MoveSkinProcess(KM.Process):
    """Moves the curves of a skin rigidly by a displacement of the time of
    time_model_part_name, [dx(t), dy(t)] as expressions of the Kratos function
    parser: every control point to where it was at the start, plus the
    displacement. Before VolumeFractionGsbmProcess, with update_each_step, in
    the same list, which then tessellates the curves where they are."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "time_model_part_name": "",
            "displacement": ["0", "0"]
        }"""))
        self.model = model
        self.settings = settings
        self.displacement = [KM.GenericFunctionUtility(settings["displacement"][i].GetString()) for i in range(2)]
        self.start = None

    def ExecuteInitializeSolutionStep(self):
        geometries = list(self.model[self.settings["model_part_name"].GetString()].Geometries)
        if self.start is None:
            self.start = [[(point.X, point.Y) for point in geometry] for geometry in geometries]
        time = self.model[self.settings["time_model_part_name"].GetString()].ProcessInfo[KM.TIME]
        dx, dy = (function.CallFunction(0.0, 0.0, 0.0, time, 0.0, 0.0, 0.0) for function in self.displacement)
        for geometry, start in zip(geometries, self.start):
            for point, (x, y) in zip(geometry, start):
                point.X, point.Y = x + dx, y + dy


class ManufacturedSourceProcess(KM.Process):
    """Assigns to the elements the source of a manufactured solution u(x, y,
    t): rho c du/dt - k lap u, with SymPy, or -k lap u alone while the
    ProcessInfo has no BDF_COEFFICIENTS, the static solve that gives the
    elliptic projection of the initial condition. Both through Kratos's
    AssignScalarFieldToElementsProcess, as the source of the static benchmarks."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "variable_name": "",
            "exact_solution": "",
            "heat_capacity": 1.0,
            "conductivity": 1.0
        }"""))
        x, y, t = sp.symbols("x y t")
        u = sp.sympify(settings["exact_solution"].GetString().replace("^", "**"), locals={"x": x, "y": y, "t": t})
        laplacian = settings["conductivity"].GetDouble() * (u.diff(x, 2) + u.diff(y, 2))
        sources = {"static": -laplacian, "transient": settings["heat_capacity"].GetDouble() * u.diff(t) - laplacian}
        self.model_part = model[settings["model_part_name"].GetString()]
        self.processes = {}
        for name, source in sources.items():
            parameters = KM.Parameters("""{"local_axes": {}}""")
            parameters.AddValue("model_part_name", settings["model_part_name"])
            parameters.AddValue("variable_name", settings["variable_name"])
            parameters.AddString("value", str(source).replace("**", "^"))
            self.processes[name] = KM.AssignScalarFieldToElementsProcess(self.model_part, parameters)

    def ExecuteInitializeSolutionStep(self):
        transient = self.model_part.ProcessInfo.Has(KM.BDF_COEFFICIENTS)
        self.processes["transient" if transient else "static"].ExecuteInitializeSolutionStep()


class BoundaryDataProcess(KM.Process):
    """Assigns what the GSBM Dirichlet conditions read: g, to the unknown, at
    the closest skin points; and, with a Kratos that has the shifted subscale
    (VOLUME_SOURCE_DERIVATIVES), f and its Cartesian derivatives of orders 0
    to p - 2, every order n as d_x^(n-k) d_y^k, k = 0 to n, at the points of
    the cut facets of facets_model_part_name, those with an IDENTIFIER. g and
    f are expressions of the Kratos function parser, f differentiated with
    SymPy. The nodes and cut facets are found when the step starts, as
    VolumeFractionGsbmProcess creates them."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "facets_model_part_name": "",
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

        x_, y_, z_, t_ = self.symbols

        def derivatives(expression, orders, points):
            """Values at the points of every d_x^(n-k) d_y^k expression, n in orders."""
            x, y = np.array([[point.X, point.Y] for point in points]).T
            columns = []
            for n in orders:
                for k in range(n + 1):
                    derivative = expression.diff(*[x_] * (n - k), *[y_] * k) if n > 0 else expression
                    values = sp.lambdify(self.symbols, derivative, "numpy")(x, y, 0., time)
                    columns.append(np.broadcast_to(values, x.shape))
            return np.array(columns).T.reshape(len(points), -1)

        for node, value in zip(nodes, derivatives(self.g, [0], nodes)[:, 0]):
            node.SetValue(self.variable, value)

        if not hasattr(IGA, "VOLUME_SOURCE_DERIVATIVES"):
            return  # no shifted subscale to feed

        # The cut facets, at their points (the centers of their - sides)
        facets = [facet for facet in self.model[self.settings["facets_model_part_name"].GetString()].Conditions
                  if facet.Has(KM.IDENTIFIER)]
        points = [facet.GetGeometry().Center() for facet in facets]
        for facet, f_row in zip(facets, derivatives(self.f, range(degree - 1), points)):
            facet.SetValue(IGA.VOLUME_SOURCE_DERIVATIVES, KM.Vector(f_row.tolist()))


class ActiveSpansProcess(KM.Process):
    """Keeps the problem on the knot spans with alpha > 0, as CutFEM does:
    fixes to zero the unknown of every basis function whose support lies only
    on empty spans (alpha = 0), which removes it, and deactivates the ghost
    penalty conditions of ghost_penalty_model_part_name unless both their
    spans have alpha > 0. Next to an empty span the ghost penalty would pull
    towards a polynomial with some of its functions removed, which is not
    consistent. Without the process, the whole background is extended by the ghost
    penalty alone, which at p >= 4 leaves the system so ill-conditioned that
    round-off sets the error. Runs when the step starts, after
    VolumeFractionGsbmProcess has assigned the fractions."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "", "geometry_name": "", "elements_model_part_name": "",
            "ghost_penalty_model_part_name": "", "variable_name": "" }"""))
        self.model = model
        self.settings = settings
        self.variable = KM.KratosGlobals.GetVariable(settings["variable_name"].GetString())

    def ExecuteInitializeSolutionStep(self):
        model_part = self.model[self.settings["model_part_name"].GetString()]
        background = model_part.GetGeometry(self.settings["geometry_name"].GetString())
        spans = [np.array(background.SpansLocalSpace(direction)) for direction in range(2)]
        alpha = knot_span_volume_fractions(
            self.model[self.settings["elements_model_part_name"].GetString()].Elements, spans)

        # The function (a, b) lives on the spans a - p to a and b - q to b
        p, q = background.PolynomialDegreeU(), background.PolynomialDegreeV()
        filled = alpha > 0.
        active = np.zeros((filled.shape[0] + p, filled.shape[1] + q), dtype=bool)
        for i in range(p + 1):
            for j in range(q + 1):
                active[i:i + filled.shape[0], j:j + filled.shape[1]] |= filled
        # Control points ordered u fastest
        for node, keep in zip(background, active.T.ravel()):
            if not keep:
                node.Fix(self.variable)
                node.SetSolutionStepValue(self.variable, 0.)

        ghost = self.model[self.settings["ghost_penalty_model_part_name"].GetString()].Conditions
        dropped = 0
        for condition in ghost:
            centers = [condition.GetGeometry().GetGeometryPart(side).Center() for side in (0, 1)]
            sides = alpha[span_indices(spans[0], np.array([c.X for c in centers])),
                          span_indices(spans[1], np.array([c.Y for c in centers]))]
            if not sides.all():
                condition.Set(KM.ACTIVE, False)
                dropped += 1
        KM.Logger.PrintInfo("ActiveSpansProcess", f"{np.count_nonzero(~active)} of {active.size} functions "
                            f"and {dropped} of {len(ghost)} ghost facets dropped")


class L2ErrorProcess(KM.Process):
    """Alpha-weighted L2 norm of the error, Eq. (48): the sum over the knot
    spans of alpha times the integral of (u - u_h)^2, with
    number_of_integration_points_per_span Gauss points per direction (0 for
    p + 1, the Gauss points of the elements); and the same norm of u, for the
    relative error. Also the L2 norms of both on whole knot spans: on the
    surrogate domain of the SBM as Hollweck et al. measure it (IGAeasy), the
    spans with alpha >= 0.5, and on the spans inside, alpha = 1, Eq. (48) with
    binary weights. With a skin_model_part_name, alpha is the volume fraction
    of the skin whatever the weights of the elements, so that all the methods
    are measured alike, and there are also the L2 norms of both in the domain alone,
    inside the skin polygon: the spans with alpha = 1 with those Gauss
    points, and the cut ones with domain_quadrature to
    domain_quadrature_depth levels. Over the time steps, the steps with
    BDF_COEFFICIENTS, also the norms in time of these of Eqs. (54) and (55),
    from time_norms."""

    def __init__(self, model, settings):
        super().__init__()
        settings.ValidateAndAssignDefaults(KM.Parameters("""{
            "model_part_name": "",
            "geometry_name": "",
            "elements_model_part_name": "",
            "variable_name": "",
            "exact_solution": "",
            "number_of_integration_points_per_span": 0,
            "skin_model_part_name": "",
            "domain_quadrature_depth": 5
        }"""))
        self.model = model
        self.settings = settings
        self.empty_skin = model.CreateModelPart("L2ErrorEmptySkin")
        self.variable = KM.KratosGlobals.GetVariable(settings["variable_name"].GetString())
        self.exact = parse_function(settings["exact_solution"].GetString())
        self.l2_error = None
        self.l2_norm = None
        self.l2_error_domain = None
        self.l2_norm_domain = None
        self.l2_error_surrogate = None
        self.l2_norm_surrogate = None
        self.l2_error_interior = None
        self.l2_norm_interior = None
        # Every time step's DELTA_TIME and errors and norms, alpha-weighted and in the domain
        self.steps = []

    def ExecuteFinalizeSolutionStep(self):
        self.measure()
        # The time steps alone, not the elliptic projection of the initial condition
        process_info = self.model[self.settings["model_part_name"].GetString()].ProcessInfo
        if process_info.Has(KM.BDF_COEFFICIENTS):
            self.steps.append((process_info[KM.DELTA_TIME], self.l2_error, self.l2_norm,
                               self.l2_error_domain, self.l2_norm_domain))

    def time_norms(self):
        """The norms in time of the errors and of u over the time steps, the L2
        norm of Eq. (54), the root of the sum of dt e_i^2, and the largest e_i
        of Eq. (55), alpha-weighted and, with a skin, in the domain."""
        if not self.steps:
            return {}
        dt, *values = (np.array(column, dtype=float) for column in zip(*self.steps))
        names = ("error", "norm", "error_domain", "norm_domain")
        norms = {}
        for name, value in zip(names, values):
            if not np.isnan(value).any():
                norms[f"l2_l2_{name}"] = np.sqrt(np.sum(dt * value**2))
                norms[f"linf_l2_{name}"] = np.max(value)
        return norms

    def measure(self):
        """The errors and norms of u at the current step."""
        model_part = self.model[self.settings["model_part_name"].GetString()]
        background = model_part.GetGeometry(self.settings["geometry_name"].GetString())
        spans = [np.array(background.SpansLocalSpace(direction)) for direction in range(2)]
        skin_name = self.settings["skin_model_part_name"].GetString()
        if skin_name:
            # Those of the skin, as with binary weights the elements hide the cut spans
            alpha = np.array(IGA.VolumeFractionUtilities.ComputeKnotSpanVolumeFractions(
                self.model[skin_name], self.empty_skin, spans[0].tolist(), spans[1].tolist()))
        else:
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
        def span_integrals(values):
            """The integral of values^2 over every knot span."""
            return (values**2 * np.outer(*weights)).reshape(
                alpha.shape[0], points_per_span[0], alpha.shape[1], points_per_span[1]).sum(axis=(1, 3))

        error_squared, u_squared = span_integrals(u - u_h), span_integrals(u)
        self.l2_error = np.sqrt(np.sum(alpha * error_squared))
        self.l2_norm = np.sqrt(np.sum(alpha * u_squared))
        KM.Logger.PrintInfo("L2ErrorProcess", f"L2 error: {self.l2_error:.6e}, "
                            f"relative: {self.l2_error / self.l2_norm:.6e}")

        # On whole knot spans: the surrogate domain of the SBM, alpha >= 0.5,
        # and the spans inside
        surrogate, interior = alpha >= .5, alpha == 1.
        self.l2_error_surrogate = np.sqrt(np.sum(error_squared[surrogate]))
        self.l2_norm_surrogate = np.sqrt(np.sum(u_squared[surrogate]))
        self.l2_error_interior = np.sqrt(np.sum(error_squared[interior]))
        self.l2_norm_interior = np.sqrt(np.sum(u_squared[interior]))

        if not skin_name:
            return
        skin = np.array([[condition.GetGeometry()[0].X, condition.GetGeometry()[0].Y]
                         for condition in self.model[skin_name].Conditions])
        cut = np.argwhere((alpha > 0.) & (alpha < 1.))
        sizes = np.diff(spans[0])[cut[:, 0]]
        if not np.allclose(sizes, np.diff(spans[1])[cut[:, 1]]):
            raise ValueError("L2ErrorProcess: the error in the domain needs square knot spans")
        points, point_weights = domain_quadrature(
            Polygon(skin), np.column_stack([spans[0][cut[:, 0]], spans[1][cut[:, 1]], sizes]),
            max(points_per_span), self.settings["domain_quadrature_depth"].GetInt())
        cut_u_h = background_point_values(background, self.variable, points)
        cut_u = np.array([self.exact(x, y, time) for x, y in points])
        inside = alpha == 1.
        self.l2_error_domain = np.sqrt(np.sum(error_squared[inside]) + np.sum(point_weights * (cut_u - cut_u_h)**2))
        self.l2_norm_domain = np.sqrt(np.sum(u_squared[inside]) + np.sum(point_weights * cut_u**2))
        KM.Logger.PrintInfo("L2ErrorProcess", f"L2 error in the domain: {self.l2_error_domain:.6e}, "
                            f"relative: {self.l2_error_domain / self.l2_norm_domain:.6e}")


class VtkFieldsOutputProcess(KM.OutputProcess):
    """Writes u_h, u_exact, error, volume_fraction and inside on a uniform
    grid of the background (u_h also outside the domain) as legacy VTK
    structured points, <output_file_name>_t<t>.vtk, and the skin polygon,
    <skin_output_file_name>_t<t>.vtk, at the start and every output_interval
    time steps after."""

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
            "output_interval": 1,
            "output_file_name": "",
            "skin_output_file_name": ""
        }"""))
        self.model = model
        self.settings = settings
        self.variable = KM.KratosGlobals.GetVariable(settings["variable_name"].GetString())
        self.exact = parse_function(settings["exact_solution"].GetString())
        self.steps = 0

    def IsOutputStep(self):
        self.steps += 1
        return (self.steps - 1) % self.settings["output_interval"].GetInt() == 0

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

        # Appended, as the time would look like a suffix
        path = Path(f'{self.settings["output_file_name"].GetString()}_t{time:.4f}.vtk')
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0\nGSBM Poisson\nASCII\nDATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {samples} {samples} 1\nORIGIN {grids[0][0]} {grids[1][0]} 0\n")
            f.write(f"SPACING {grids[0][1] - grids[0][0]} {grids[1][1] - grids[1][0]} 1\n")
            f.write(f"POINT_DATA {samples * samples}\n")
            # VTK orders the points with x fastest, so the [x, y] arrays go transposed
            for name, field in (("u_h", u_h), ("u_exact", u), ("error", u_h - u),
                                ("volume_fraction", alpha), ("inside", inside * 1.)):
                f.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
                np.savetxt(f, field.T.reshape(-1), fmt="%.10e")

        skin_path = Path(f'{self.settings["skin_output_file_name"].GetString()}_t{time:.4f}.vtk')
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
