"""
Convergence study: every case on N x N knot spans (N = 20 to 320 by default)
at every given degree p (by default the one of its json), with the other
settings of its ProjectParameters json and without the field output. Writes
the L2 error of Eq. (48), the relative one, ||u - u_h|| / ||u||, and the rate
to out/convergence_p<p>.csv, replacing only the cases run, and the Kratos log
to out/convergence_p<p>.log. All cases by default:

    python3 convergence.py [config/ProjectParameters_<case>.json ...] [--degrees 1 2 3 4]
        [--sizes 40 80 160 320] [--materials "config/MaterialParameters_p{p}.json"]

--materials takes the materials of every degree instead of those of the json,
with {p} for the degree. figures.py plots the results.
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import KratosMultiphysics as KM

from gsbm_poisson_analysis import GsbmPoissonAnalysis

SIZES = (20, 40, 80, 160, 320)

# WSBM curves of Fig. 7 (Q1 elements), read off the plot to about 10 %
PAPER_WSBM = {
    "circle": dict(zip(SIZES, (0.16, 0.058, 0.016, 0.0040, 0.0010))),
    "flower": dict(zip(SIZES, (0.33, 0.14, 0.063, 0.012, 0.0025))),
    "parallelogram": dict(zip(SIZES, (0.17, 0.051, 0.015, 0.0041, 0.0012))),
}


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


def l2_errors(parameters, size, log):
    """L2 error and L2 norm of u of a case on size x size knot spans."""
    parameters = parameters.Clone()
    parameters.RemoveValue("output_processes")
    background_parameters(parameters)["number_of_knot_spans"] = KM.Parameters(f"[{size}, {size}]")
    analysis = GsbmPoissonAnalysis(KM.Model(), parameters)
    with logged_to(log):
        analysis.Run()
    data = analysis.GetFinalData()
    return data["l2_error"], data["l2_norm"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="ProjectParameters json files (all in config/)")
    parser.add_argument("--degrees", nargs="+", type=int, help="polynomial degrees (the one of every json)")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES, help="knot spans per direction")
    parser.add_argument("--materials", help="materials of every degree, with {p} for the degree")
    arguments = parser.parse_args()

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
        log = Path(f"out/convergence_p{degree}.log")
        log.write_text("")
        rows = ["case,N,l2_error,relative_error,rate,paper_wsbm,time_s"]
        cases_run = set()
        for parameters in cases:
            parameters = parameters.Clone()
            set_degree(parameters, degree)
            if arguments.materials:
                set_materials(parameters, arguments.materials, degree)
            case = parameters["problem_data"]["problem_name"].GetString()
            # The paper's WSBM curves are for Q1 elements
            paper = PAPER_WSBM[case.split("_")[0]] if degree == 1 else {}
            cases_run.add(case)
            print(f"\n{case}, p = {degree}")
            print(f"{'N':>5} {'L2 error':>11} {'relative':>11} {'rate':>6} {'paper':>9} {'time [s]':>9}")
            previous = None
            for size in arguments.sizes:
                start = time.perf_counter()
                error, norm = l2_errors(parameters, size, log)
                elapsed = time.perf_counter() - start
                rate = np.log(previous / error) / np.log(size / previous_size) if previous else float("nan")
                previous, previous_size = error, size
                print(f"{size:>5} {error:>11.3e} {error / norm:>11.3e} {rate:>6.2f} "
                      f"{paper.get(size, ''):>9} {elapsed:>9.1f}", flush=True)
                rows.append(f"{case},{size},{error:.6e},{error / norm:.6e},{rate:.3f},"
                            f"{paper.get(size, '')},{elapsed:.2f}")

        # Keep the results of the other cases
        path = Path(f"out/convergence_p{degree}.csv")
        if path.exists():
            rows[1:1] = [row for row in path.read_text().splitlines()[1:] if row.split(",")[0] not in cases_run]
        path.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
