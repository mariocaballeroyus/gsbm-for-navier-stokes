"""
Runs cases of the GSBM Poisson benchmark from their ProjectParameters json,
all of them by default; with --degrees, at each of those degrees, and with
--cells, on that many knot spans per direction, with the field outputs named
<output_file_name>_p<p>_N<N>; with --materials, with the materials of every
degree, {p} standing for it:

    python3 MainKratos.py [config/ProjectParameters_<case>.json ...] [--degrees 1 2 3 4]
        [--cells 80] [--materials "config/MaterialParameters_p{p}.json"]
"""

import argparse
import os
from pathlib import Path

import KratosMultiphysics as KM

from convergence import background_parameters, set_degree, set_materials
from gsbm_poisson_analysis import GsbmPoissonAnalysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="ProjectParameters json files (all in config/)")
    parser.add_argument("--degrees", nargs="+", type=int, help="polynomial degrees (the one of every json)")
    parser.add_argument("--cells", type=int, help="knot spans per direction (those of every json)")
    parser.add_argument("--materials", help="materials of every degree, with {p} for the degree")
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
            if arguments.materials:
                set_materials(parameters, arguments.materials, background["polynomial_order"][0].GetInt())

            # Outputs named by degree and mesh when they are not the json's
            if degree is not None or arguments.cells is not None:
                p, n = background["polynomial_order"][0].GetInt(), background["number_of_knot_spans"][0].GetInt()
                for output in parameters["output_processes"]["vtk_output"].values():
                    name = output["Parameters"]["output_file_name"]
                    name.SetString(f"{name.GetString()}_p{p}_N{n}")
            GsbmPoissonAnalysis(KM.Model(), parameters).Run()
