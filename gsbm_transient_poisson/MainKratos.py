"""
Runs cases of the GSBM transient Poisson benchmark on a moving domain from their
ProjectParameters json, all of them by default, and prints the errors at the
last step and in time; with --degrees, at each of those degrees, with --cells,
on that many knot spans per direction, with --time-step and --end-time, with
those, and with --materials, with the materials of every degree, {p} standing
for it. The errors of every step go to
<output>/<case>_p<p>_N<N>_dt<dt>.csv (out by default) as the step finishes, and
with --fields, the fields every that many steps to <output>/<case>_p<p>_t<t>.vtk:

    python3 MainKratos.py [config/ProjectParameters_<case>.json ...] [--degrees 1 2 3]
        [--cells 100] [--time-step 0.005] [--end-time 1]
        [--materials "config/MaterialParameters_skew_p{p}.json"] [--output out] [--fields 5]
"""

import argparse
import json
import os
from pathlib import Path

import KratosMultiphysics as KM

from gsbm_transient_poisson_analysis import GsbmTransientPoissonAnalysis


def background_parameters(parameters):
    """The settings of the background patch."""
    return next(modeler["Parameters"] for modeler in parameters["modelers"].values()
                if modeler["modeler_name"].GetString() == "NurbsGeometryModeler")


def set_degree(parameters, degree):
    """B-splines of degree p on the background, and the p-th derivatives that
    the GSBM facets need."""
    background_parameters(parameters)["polynomial_order"] = KM.Parameters(f"[{degree}, {degree}]")
    for modeler in parameters["modelers"].values():
        if modeler["modeler_name"].GetString() == "IgaModelerGsbm":
            for entry in modeler["Parameters"]["element_condition_list"].values():
                if entry.Has("gsbm_parameters") and entry.Has("shape_function_derivatives_order"):
                    entry["shape_function_derivatives_order"].SetInt(degree + 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="ProjectParameters json files (all in config/)")
    parser.add_argument("--degrees", nargs="+", type=int, help="polynomial degrees (the one of every json)")
    parser.add_argument("--cells", type=int, help="knot spans per direction (those of every json)")
    parser.add_argument("--time-step", type=float, help="time step (that of every json)")
    parser.add_argument("--end-time", type=float, help="end time (that of every json)")
    parser.add_argument("--materials", help="materials of every degree, with {p} for the degree")
    parser.add_argument("--output", type=Path, default=Path("out"), help="folder of the errors of every step")
    parser.add_argument("--fields", type=int, help="write the fields every that many steps")
    arguments = parser.parse_args()

    # The json files give the skins, materials and outputs relative to this folder
    here = Path(__file__).resolve().parent
    files = [file.resolve() for file in arguments.files] or sorted(here.glob("config/ProjectParameters_*.json"))
    os.chdir(here)
    for file in files:
        for degree in arguments.degrees or [None]:
            parameters = KM.Parameters(file.read_text())
            background = background_parameters(parameters)
            if degree is not None:
                set_degree(parameters, degree)
            if arguments.cells is not None:
                background["number_of_knot_spans"] = KM.Parameters(f"[{arguments.cells}, {arguments.cells}]")
            if arguments.time_step is not None:
                parameters["solver_settings"]["time_stepping"]["time_step"].SetDouble(arguments.time_step)
            if arguments.end_time is not None:
                parameters["problem_data"]["end_time"].SetDouble(arguments.end_time)
            if arguments.materials:
                parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
                    arguments.materials.format(p=background["polynomial_order"][0].GetInt()))

            p, n = background["polynomial_order"][0].GetInt(), background["number_of_knot_spans"][0].GetInt()
            case = file.stem.removeprefix("ProjectParameters_")
            time_step = parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble()
            arguments.output.mkdir(parents=True, exist_ok=True)
            history = arguments.output / f"{case}_p{p}_N{n}_dt{time_step:g}.csv"
            if arguments.fields:
                error = next(process["Parameters"] for process in parameters["processes"]["auxiliar_process_list"].values())
                fields = KM.Parameters(json.dumps([{
                    "python_module": "gsbm_processes",
                    "process_name": "VtkFieldsOutputProcess",
                    "Parameters": {
                        "model_part_name": "IgaModelPart",
                        "geometry_name": "background",
                        "elements_model_part_name": "IgaModelPart.Domain",
                        "skin_model_part_name": "Skin.tessellation",
                        "variable_name": "TEMPERATURE",
                        "exact_solution": error["exact_solution"].GetString(),
                        "number_of_samples": 201,
                        "output_interval": arguments.fields,
                        "output_file_name": str(arguments.output / f"{case}_p{p}"),
                        "skin_output_file_name": str(arguments.output / f"skin_{case}_p{p}")
                    }
                }]))
                parameters["output_processes"].AddValue("vtk_output", fields)
            analysis = GsbmTransientPoissonAnalysis(KM.Model(), parameters, history)
            analysis.Run()
            print(json.dumps({"case": case, "degree": p, "cells": n,
                              "time_step": parameters["solver_settings"]["time_stepping"]["time_step"].GetDouble(),
                              **{key: float(value) if value is not None else None
                                 for key, value in analysis.GetFinalData().items()}}), flush=True)
