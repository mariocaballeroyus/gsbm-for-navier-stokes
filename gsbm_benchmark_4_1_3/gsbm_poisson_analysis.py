"""
Analysis stage of the GSBM Poisson benchmark, driven by a ProjectParameters
json: the modelers build the background patch, the GSBM entities and the skin;
the processes the volume fractions, the closest skin points, u_D and f.

The solver is a linear static one for the Laplacian entities of IgaApplication.
ConvectionDiffusionApplication is not built, so it takes the place of its
stationary solver, with the same names for the settings it uses.
"""

import KratosMultiphysics as KM
import KratosMultiphysics.IgaApplication  # noqa: F401 (GSBM modeler, entities and processes)
import KratosMultiphysics.LinearSolversApplication  # noqa: F401 (sparse_lu)
from KratosMultiphysics.analysis_stage import AnalysisStage
from KratosMultiphysics.python_linear_solver_factory import ConstructSolver
from KratosMultiphysics.python_solver import PythonSolver

from gsbm_processes import L2ErrorProcess


class GsbmPoissonSolver(PythonSolver):
    """Adds the variables and DOFs, reads the materials and solves the linear
    system once per step. The elements and conditions come from the modelers."""

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
                "gradient_variable": "TEMPERATURE_GRADIENT"
            },
            "time_stepping": {
                "time_step": 1.0
            },
            "linear_solver_settings": {
                "solver_type": "sparse_lu"
            }
        }""")

    def __init__(self, model, settings):
        super().__init__(model, settings)
        self.model_part = model.CreateModelPart(self.settings["model_part_name"].GetString())
        self.model_part.ProcessInfo[KM.DOMAIN_SIZE] = self.settings["domain_size"].GetInt()

        variables = self.settings["convection_diffusion_variables"]
        self.unknown, self.diffusion, self.volume_source, gradient = (
            KM.KratosGlobals.GetVariable(variables[name].GetString())
            for name in ("unknown_variable", "diffusion_variable",
                         "volume_source_variable", "gradient_variable"))
        self.convection_diffusion_settings = KM.ConvectionDiffusionSettings()
        self.convection_diffusion_settings.SetUnknownVariable(self.unknown)
        self.convection_diffusion_settings.SetDiffusionVariable(self.diffusion)
        self.convection_diffusion_settings.SetVolumeSourceVariable(self.volume_source)
        self.convection_diffusion_settings.SetGradientVariable(gradient)

    def AddVariables(self):
        for variable in (self.unknown, self.diffusion, self.volume_source):
            self.model_part.AddNodalSolutionStepVariable(variable)

    def ImportModelPart(self):
        self._ImportModelPart(self.model_part, self.settings["model_import_settings"])

    def PrepareModelPart(self):
        self.model_part.ProcessInfo.SetValue(
            KM.CONVECTION_DIFFUSION_SETTINGS, self.convection_diffusion_settings)
        materials = KM.Parameters('{"Parameters": {}}')
        materials["Parameters"].AddValue(
            "materials_filename", self.settings["material_import_settings"]["materials_filename"])
        KM.ReadMaterialsUtility(materials, self.model)

    def AddDofs(self):
        KM.VariableUtils().AddDof(self.unknown, self.model_part)

    def GetMinimumBufferSize(self):
        return 1

    def GetComputingModelPart(self):
        return self.model_part

    def AdvanceInTime(self, current_time):
        new_time = current_time + self.settings["time_stepping"]["time_step"].GetDouble()
        self.model_part.ProcessInfo[KM.STEP] += 1
        self.model_part.CloneTimeStep(new_time)
        return new_time

    def Initialize(self):
        linear_solver = ConstructSolver(self.settings["linear_solver_settings"])
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


class GsbmPoissonAnalysis(AnalysisStage):

    def _CreateSolver(self):
        return GsbmPoissonSolver(self.model, self.project_parameters["solver_settings"])

    def _GetOrderOfProcessesInitialization(self):
        # The GSBM processes set the volume fractions and the closest skin
        # points, which the others use (the lists are otherwise in key order)
        return ["gsbm_process_list", "boundary_conditions_process_list",
                "source_process_list", "auxiliar_process_list"]

    def _GetSimulationName(self):
        return "GSBM Poisson Analysis"

    def GetFinalData(self):
        """The L2 error and the L2 norm of u, if an L2ErrorProcess computed them."""
        data = {}
        for process in self._GetListOfProcesses():
            if isinstance(process, L2ErrorProcess):
                data.update(l2_error=process.l2_error, l2_norm=process.l2_norm)
        return data
