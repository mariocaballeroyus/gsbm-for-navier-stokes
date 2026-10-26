"""
Runs the colliding circles from their ProjectParameters json, all cases by
default, at positions d of the moving circle (every d of skins.py by
default, as figures.py animates them); with --degrees, at each of those degrees; with --cells, on that many
knot spans per direction; with --materials, with the materials of every
degree, {p} standing for it. The field outputs are named
<output_file_name>_p<p>_d<d>, and the skins <skin_output_file_name>_d<d>;
with several positions, <output_file_name>_p<p>.vtk.series and
<skin_output_file_name>.vtk.series list them with d as the time, for ParaView
to play them as one animation:

    python3 MainKratos.py [config/ProjectParameters_<case>.json ...] [--d -0.1 0 0.1]
        [--degrees 1 2 3 4 5] [--cells 100] [--materials "config/MaterialParameters_p{p}.json"]
"""

import argparse
import json
import os
from pathlib import Path

import KratosMultiphysics as KM

from gsbm_poisson_analysis import GsbmPoissonAnalysis
from skins import D_VALUES, position_name
from sweep import background_parameters, set_cells, set_degree, set_materials, set_position

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="ProjectParameters json files (all in config/)")
    parser.add_argument("--d", nargs="+", type=float, default=D_VALUES,
                        help="positions of the moving circle (every d of skins.py)")
    parser.add_argument("--degrees", nargs="+", type=int, help="polynomial degrees (the one of every json)")
    parser.add_argument("--cells", type=int, help="knot spans per direction (those of every json)")
    parser.add_argument("--materials", help="materials of every degree, with {p} for the degree")
    arguments = parser.parse_args()

    # The json files give the skins, materials and outputs relative to this folder
    here = Path(__file__).resolve().parent
    files = [file.resolve() for file in arguments.files] or sorted(here.glob("config/ProjectParameters_*.json"))
    os.chdir(here)
    series = set()  # the names of the outputs written by position
    for file in files:
        for d in arguments.d:
            for degree in arguments.degrees or [None]:
                parameters = KM.Parameters(file.read_text())
                background = background_parameters(parameters)
                if d is not None:
                    set_position(parameters, d)
                if degree is not None:
                    set_degree(parameters, degree)
                if arguments.cells is not None:
                    set_cells(parameters, arguments.cells)
                if arguments.materials:
                    set_materials(parameters, arguments.materials, background["polynomial_order"][0].GetInt())

                # Outputs named by degree and position when they are not the json's
                if d is not None or degree is not None:
                    p = background["polynomial_order"][0].GetInt()
                    position = position_name(d if d is not None else 0.)
                    for output in parameters["output_processes"]["vtk_output"].values():
                        names = output["Parameters"]
                        for key, name in (("output_file_name", f"{names['output_file_name'].GetString()}_p{p}"),
                                          ("skin_output_file_name", names["skin_output_file_name"].GetString())):
                            names[key].SetString(f"{name}_d{position}")
                            series.add(name)
                GsbmPoissonAnalysis(KM.Model(), parameters).Run()

    # ParaView's file series, by position: every position on disk, from this
    # run or earlier ones
    for name in series:
        name = Path(name)
        members = {float(path.name[len(name.name) + 2:-len(".vtk")]): path.name
                   for path in name.parent.glob(f"{name.name}_d*.vtk")}
        if len(members) > 1:
            files = [{"name": member, "time": d} for d, member in sorted(members.items())]
            Path(f"{name}.vtk.series").write_text(
                json.dumps({"file-series-version": "1.0", "files": files}, indent=2) + "\n")
