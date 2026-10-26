"""
Sweep over the position d of the moving circle: every case at every d of
skins.py (or the given ones, or those of a range) on N x N knot spans (100 by
default, as in the paper), at every given degree p (by default the one of its
json), with the other settings of its ProjectParameters json and without the
field output. Writes the L2 error of Eq. (48) and the relative one,
||u - u_h|| / ||u||, to <output>_p<p>.csv (out/sweep by default) after every
run, replacing only the cases and positions run, and appends the Kratos log to
<output>_p<p>.log. All cases by default:

    python3 sweep.py [config/ProjectParameters_<case>.json ...] [--degrees 1 2 3 4 5]
        [--d -0.1 0 0.1 | --d-range 0.1 0.102 0.00002] [--cells 100]
        [--materials "config/MaterialParameters_p{p}.json"] [--output out/sweep]

--materials takes the materials of every degree instead of those of the json,
with {p} for the degree. figures.py plots the results.
"""

import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import KratosMultiphysics as KM

from gsbm_poisson_analysis import GsbmPoissonAnalysis
from skins import D_VALUES, ensure_skin

CELLS = 100
COLUMNS = ("case", "d", "l2_error", "relative_error", "time_s")


@contextmanager
def logged_to(path):
    """Sends the standard output, where Kratos logs, to the end of a file."""
    KM.Logger.Flush()
    sys.stdout.flush()
    terminal = os.dup(1)
    with open(path, "a") as log:
        os.dup2(log.fileno(), 1)
    try:
        yield
    finally:
        KM.Logger.Flush()
        sys.stdout.flush()
        os.dup2(terminal, 1)
        os.close(terminal)


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


def set_materials(parameters, materials, degree):
    """The materials of the degree, from a file name with {p} for it."""
    parameters["solver_settings"]["material_import_settings"]["materials_filename"].SetString(
        materials.format(p=degree))


def set_cells(parameters, cells):
    """cells x cells knot spans on the background."""
    background_parameters(parameters)["number_of_knot_spans"] = KM.Parameters(f"[{cells}, {cells}]")


def set_position(parameters, d):
    """The skin of the circles at a position d of the moving one."""
    for modeler in parameters["modelers"].values():
        if modeler["modeler_name"].GetString() == "ImportNurbsSbmModeler":
            modeler["Parameters"]["input_filename"].SetString(ensure_skin(d))


def l2_errors(parameters, log):
    """L2 error and L2 norm of u of a case."""
    parameters = parameters.Clone()
    parameters.RemoveValue("output_processes")
    analysis = GsbmPoissonAnalysis(KM.Model(), parameters)
    with logged_to(log):
        analysis.Run()
    data = analysis.GetFinalData()
    return data["l2_error"], data["l2_norm"]


def write_results(path, results):
    """Merges results, keyed by case and position, into a CSV file: they
    replace the rows of the same case and position and keep the others, every
    case in the order of the file and by position."""
    rows = {}
    if path.exists():
        with open(path) as f:
            rows = {(row["case"], float(row["d"])): row for row in csv.DictReader(f)}
    rows.update(results)
    lines = [",".join(COLUMNS)]
    for case in dict.fromkeys(case for case, _ in rows):
        for d in sorted(d for c, d in rows if c == case):
            lines.append(",".join(str(rows[(case, d)][column]) for column in COLUMNS))
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="ProjectParameters json files (all in config/)")
    parser.add_argument("--degrees", nargs="+", type=int, help="polynomial degrees (the one of every json)")
    parser.add_argument("--d", nargs="+", type=float, default=D_VALUES, help="positions of the moving circle")
    parser.add_argument("--d-range", nargs=3, type=float, metavar=("START", "STOP", "STEP"),
                        help="positions from START to STOP, both included, every STEP")
    parser.add_argument("--cells", type=int, default=CELLS, help="knot spans per direction")
    parser.add_argument("--materials", help="materials of every degree, with {p} for the degree")
    parser.add_argument("--output", default="out/sweep", help="results as <output>_p<p>.csv")
    arguments = parser.parse_args()
    if arguments.d_range:
        start, stop, step = arguments.d_range
        arguments.d = [round(start + k * step, 6) for k in range(int(round((stop - start) / step)) + 1)]

    # The json files give the skins, materials and outputs relative to this folder
    here = Path(__file__).resolve().parent
    files = [file.resolve() for file in arguments.files] or sorted(here.glob("config/ProjectParameters_*.json"))
    os.chdir(here)

    # The cases to run at every degree
    runs = {}
    for file in files:
        parameters = KM.Parameters(file.read_text())
        own_degree = background_parameters(parameters)["polynomial_order"][0].GetInt()
        for degree in arguments.degrees or [own_degree]:
            runs.setdefault(degree, []).append(parameters)

    for degree, cases in sorted(runs.items()):
        log, path = Path(f"{arguments.output}_p{degree}.log"), Path(f"{arguments.output}_p{degree}.csv")
        for parameters in cases:
            parameters = parameters.Clone()
            set_degree(parameters, degree)
            set_cells(parameters, arguments.cells)
            if arguments.materials:
                set_materials(parameters, arguments.materials, degree)
            case = parameters["problem_data"]["problem_name"].GetString()
            print(f"\n{case}, p = {degree}, N = {arguments.cells}")
            print(f"{'d':>8} {'L2 error':>11} {'relative':>11} {'time [s]':>9}")
            for d in arguments.d:
                set_position(parameters, d)
                start = time.perf_counter()
                error, norm = l2_errors(parameters, log)
                elapsed = time.perf_counter() - start
                print(f"{d:>8.5f} {error:>11.3e} {error / norm:>11.3e} {elapsed:>9.1f}", flush=True)
                write_results(path, {(case, d): {"case": case, "d": f"{d:.5f}", "l2_error": f"{error:.6e}",
                                                 "relative_error": f"{error / norm:.6e}",
                                                 "time_s": f"{elapsed:.2f}"}})


if __name__ == "__main__":
    main()
