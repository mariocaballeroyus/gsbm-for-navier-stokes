"""
Analysis stage of the GSBM transient Poisson benchmark, rho c du/dt - k lap u = f, on a
domain that may move, driven by a ProjectParameters json: the modelers build
the background patch, the GSBM entities and the skin; the processes move the
skin, recompute the volume fractions and closest skin points every step, and
set u_D and f.

The time derivative is GsbmLaplacianElement's BDF term, which it adds while
the ProcessInfo has BDF_COEFFICIENTS, as the fluid solvers do it: the solver
computes them every step with Kratos's TimeDiscretization.BDF, of order 1 at
the first step and time_order after. Before the first step, a static solve at
the start time, without them, gives the elliptic projection of the initial
condition (ManufacturedSourceProcess assigns -k lap u then), as the values at
the control points cannot be the solution's.

The solver is a linear one for the Laplacian entities of IgaApplication.
ConvectionDiffusionApplication is not built, so it takes the place of its
transient solver, with the same names for the settings it uses.
"""

import KratosMultiphysics as KM
import KratosMultiphysics.IgaApplication  # noqa: F401 (GSBM modeler, entities and processes)
import KratosMultiphysics.LinearSolversApplication  # noqa: F401 (sparse_lu)
from KratosMultiphysics.analysis_stage import AnalysisStage
from KratosMultiphysics.python_linear_solver_factory import ConstructSolver
from KratosMultiphysics.python_solver import PythonSolver

from gsbm_processes import L2ErrorProcess


class GsbmTransientPoissonSolver(PythonSolver):
    """Adds the variables and DOFs, reads the materials and solves the linear
    system once per step, with the BDF coefficients of the step. The elements
    and conditions come from the modelers."""

    @classmethod
    def GetDefaultParameters(cls):
        return KM.Parameters("""{
            "model_part_name": "",
            "domain_size": 2,
            "echo_level": 0,
            "model_import_settings": {
                "input_type": "use_input_model_part"
            },
            "material_import_settings": {
                "materials_filename": ""
            },
            "convection_diffusion_variables": {
                "unknown_variable": "TEMPERATURE",
                "diffusion_variable": "CONDUCTIVITY",
                "volume_source_variable": "HEAT_FLUX",
                "gradient_variable": "TEMPERATURE_GRADIENT",
                "density_variable": "DENSITY",
                "specific_heat_variable": "SPECIFIC_HEAT"
            },
            "time_stepping": {
                "time_step": 0.1
            },
            "time_order": 2,
            "linear_solver_settings": {
                "solver_type": "sparse_lu"
            }
        }""")

    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.model_part = model.CreateModelPart(self.settings["model_part_name"].GetString())
        self.model_part.ProcessInfo[KM.DOMAIN_SIZE] = self.settings["domain_size"].GetInt()

        variables = self.settings["convection_diffusion_variables"]
        self.unknown, self.diffusion, self.volume_source, gradient, density, specific_heat = (
            KM.KratosGlobals.GetVariable(variables[name].GetString())
            for name in ("unknown_variable", "diffusion_variable", "volume_source_variable",
                         "gradient_variable", "density_variable", "specific_heat_variable"))
        self.convection_diffusion_settings = KM.ConvectionDiffusionSettings()
        self.convection_diffusion_settings.SetUnknownVariable(self.unknown)
        self.convection_diffusion_settings.SetDiffusionVariable(self.diffusion)
        self.convection_diffusion_settings.SetVolumeSourceVariable(self.volume_source)
        self.convection_diffusion_settings.SetGradientVariable(gradient)
        self.convection_diffusion_settings.SetDensityVariable(density)
        self.convection_diffusion_settings.SetSpecificHeatVariable(specific_heat)
        self.time_order = self.settings["time_order"].GetInt()
        self.time_steps = 0

    def AddVariables(self):
        for variable in (self.unknown, self.diffusion, self.volume_source):
            self.model_part.AddNodalSolutionStepVariable(variable)

    def ImportModelPart(self):
        self._ImportModelPart(self.model_part, self.settings["model_import_settings"])

    def PrepareModelPart(self):
        # The steps the BDF term reads
        self.model_part.SetBufferSize(self.GetMinimumBufferSize())
        self.model_part.ProcessInfo.SetValue(
            KM.CONVECTION_DIFFUSION_SETTINGS, self.convection_diffusion_settings)
        materials = KM.Parameters('{"Parameters": {}}')
        materials["Parameters"].AddValue(
            "materials_filename", self.settings["material_import_settings"]["materials_filename"])
        KM.ReadMaterialsUtility(materials, self.model)

    def AddDofs(self):
        KM.VariableUtils().AddDof(self.unknown, self.model_part)

    def GetMinimumBufferSize(self):
        # The current step and the time_order before it
        return self.time_order + 1

    def GetComputingModelPart(self):
        return self.model_part

    def AdvanceInTime(self, current_time):
        new_time = current_time + self.settings["time_stepping"]["time_step"].GetDouble()
        self.model_part.ProcessInfo[KM.STEP] += 1
        self.model_part.CloneTimeStep(new_time)

        # BDF of order 1 at the first step, with u^0 alone, and time_order after
        self.time_steps += 1
        KM.TimeDiscretization.BDF(min(self.time_steps, self.time_order)).ComputeAndSaveBDFCoefficients(
            self.model_part.ProcessInfo)
        return new_time

    def Initialize(self):
        # With "scaling": true, the solver of the other settings solves the
        # system scaled symmetrically by the square roots of its row norms
        settings = self.settings["linear_solver_settings"].Clone()
        scaling = settings.Has("scaling") and settings["scaling"].GetBool()
        if settings.Has("scaling"):
            settings.RemoveValue("scaling")
        linear_solver = ConstructSolver(settings)
        if scaling:
            linear_solver = KM.ScalingSolver(linear_solver, True)
        self.strategy = KM.ResidualBasedLinearStrategy(
            self.model_part, KM.ResidualBasedIncrementalUpdateStaticScheme(),
            KM.ResidualBasedBlockBuilderAndSolver(linear_solver), False, False, False, False)
        self.strategy.SetEchoLevel(self.echo_level)
        self.strategy.Initialize()

    def Check(self):
        self.strategy.Check()

    def Predict(self):
        self.strategy.Predict()

    def InitializeSolutionStep(self):
        self.strategy.InitializeSolutionStep()

    def SolveSolutionStep(self):
        return self.strategy.SolveSolutionStep()

    def FinalizeSolutionStep(self):
        self.strategy.FinalizeSolutionStep()

    def Clear(self):
        self.strategy.Clear()


class GsbmTransientPoissonAnalysis(AnalysisStage):
    """With a history_path, also appends the errors of every time step to that
    CSV file as the step finishes, so that a run stopped halfway keeps them."""

    HISTORY_COLUMNS = ("time", "dt", "l2_error", "l2_norm", "l2_error_domain", "l2_norm_domain", "relative_error",
                       "relative_error_domain", "relative_error_interior")

    def __init__(self, model, project_parameters, history_path=None):
        super().__init__(model, project_parameters)
        self.history_path = history_path
        self.steps_written = 0

    def FinalizeSolutionStep(self):
        super().FinalizeSolutionStep()
        if self.history_path is None:
            return
        for process in self._GetListOfProcesses():
            if isinstance(process, L2ErrorProcess) and len(process.steps) > self.steps_written:
                dt, error, norm, error_domain, norm_domain = process.steps[-1]
                row = (self._GetSolver().GetComputingModelPart().ProcessInfo[KM.TIME], dt, error, norm, error_domain,
                       norm_domain, error / norm, error_domain / norm_domain,
                       process.l2_error_interior / process.l2_norm_interior)
                with open(self.history_path, "w" if self.steps_written == 0 else "a") as f:
                    if self.steps_written == 0:
                        f.write(",".join(self.HISTORY_COLUMNS) + "\n")
                    f.write(",".join(f"{value:.10e}" for value in row) + "\n")
                self.steps_written = len(process.steps)

    def _CreateSolver(self):
        return GsbmTransientPoissonSolver(self.model, self.project_parameters["solver_settings"])

    def _GetOrderOfProcessesInitialization(self):
        # The GSBM processes move the skin and set the volume fractions and the
        # closest skin points, which the others use (the lists are otherwise in
        # key order)
        return ["gsbm_process_list", "boundary_conditions_process_list",
                "source_process_list", "auxiliar_process_list"]

    def _GetSimulationName(self):
        return "GSBM Transient Poisson Analysis"

    def Initialize(self):
        super().Initialize()

        # The elliptic projection of the initial condition: a static solve at
        # the start time, before the solver sets BDF_COEFFICIENTS
        self.InitializeSolutionStep()
        self.SolveSolutionStep()
        self.FinalizeSolutionStep()
        self.OutputSolutionStep()

    def GetFinalData(self):
        """The L2 error and the L2 norm of u at the last step, alpha-weighted,
        in the domain, on the surrogate domain and on the spans inside, if an
        L2ErrorProcess computed them, and their norms in time over the steps,
        Eqs. (54) and (55)."""
        data = {}
        for process in self._GetListOfProcesses():
            if isinstance(process, L2ErrorProcess):
                data.update(l2_error=process.l2_error, l2_norm=process.l2_norm,
                            l2_error_domain=process.l2_error_domain, l2_norm_domain=process.l2_norm_domain,
                            l2_error_surrogate=process.l2_error_surrogate,
                            l2_norm_surrogate=process.l2_norm_surrogate,
                            l2_error_interior=process.l2_error_interior,
                            l2_norm_interior=process.l2_norm_interior,
                            time_steps=len(process.steps), **process.time_norms())
        return data
