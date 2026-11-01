"""
Convergence study: every case on N x N knot spans (N = 20 to 320 by default)
at every given degree p (by default the one of its json), with the other
settings of its ProjectParameters json and without the field output. Writes
the L2 error of Eq. (48), the relative one, ||u - u_h|| / ||u||, the rate
and, if the json's L2ErrorProcess has a skin, the relative L2 error in the
domain alone to out/convergence_p<p>.csv after every run, replacing only the cases and
meshes run, and appends the Kratos log to out/convergence_p<p>.log. All
cases by default:

    python3 convergence.py [config/ProjectParameters_<case>.json ...] [--degrees 1 2 3 4]
        [--sizes 40 80 160 320] [--materials "config/MaterialParameters_skew_p{p}.json"]

--materials takes the materials of every degree instead of those of the json,
with {p} for the degree. figures.py plots the results.
"""

import argparse
import csv
import fcntl
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import KratosMultiphysics as KM

from gsbm_poisson_analysis import GsbmPoissonAnalysis

SIZES = (20, 40, 80, 160, 320)
COLUMNS = ("case", "N", "l2_error", "relative_error", "rate", "paper_wsbm", "time_s", "relative_error_domain")

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
    the GSBM facets need (the shift, of any order, takes the derivatives of
    the span's bases from Kratos itself)."""
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
    """L2 error and L2 norm of u of a case on size x size knot spans, and the
    relative L2 error in the domain, None without a skin."""
    parameters = parameters.Clone()
    parameters.RemoveValue("output_processes")
    background_parameters(parameters)["number_of_knot_spans"] = KM.Parameters(f"[{size}, {size}]")
    analysis = GsbmPoissonAnalysis(KM.Model(), parameters)
    with logged_to(log):
        analysis.Run()
    data = analysis.GetFinalData()
    domain = data["l2_error_domain"] / data["l2_norm_domain"] if data["l2_error_domain"] is not None else None
    return data["l2_error"], data["l2_norm"], domain


def write_result(path, result):
    """Merges a result, keyed by case and mesh, into a CSV file: it replaces
    the row of the same case and mesh and keeps the others, and the rates
    between successive meshes of every case are recomputed. Several runs can
    write at once: each merges under a lock, and replaces the file whole, so
    that a run killed while writing leaves the previous file."""
    with open(path.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = {}
        if path.exists():
            with open(path) as f:
                rows = {(row["case"], int(row["N"])): row for row in csv.DictReader(f)}
        rows[(result["case"], int(result["N"]))] = result
        lines = [",".join(COLUMNS)]
        for case in dict.fromkeys(case for case, _ in rows):  # in the order of the file
            previous = None
            for size in sorted(n for c, n in rows if c == case):
                row = rows[(case, size)]
                error = float(row["l2_error"])
                row["rate"] = f"{np.log(previous[1] / error) / np.log(size / previous[0]):.3f}" if previous else "nan"
                previous = (size, error)
                lines.append(",".join(str(row.get(column) or "") for column in COLUMNS))
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text("\n".join(lines) + "\n")
        os.replace(temporary, path)


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
        log, path = Path(f"out/convergence_p{degree}.log"), Path(f"out/convergence_p{degree}.csv")
        for parameters in cases:
            parameters = parameters.Clone()
            set_degree(parameters, degree)
            if arguments.materials:
                set_materials(parameters, arguments.materials, degree)
            case = parameters["problem_data"]["problem_name"].GetString()
            # The paper's WSBM curves are for Q1 elements, on its domains
            paper = PAPER_WSBM.get(case.split("_")[0], {}) if degree == 1 else {}
            print(f"\n{case}, p = {degree}")
            print(f"{'N':>5} {'L2 error':>11} {'relative':>11} {'in domain':>11} {'paper':>9} {'time [s]':>9}")
            for size in arguments.sizes:
                start = time.perf_counter()
                error, norm, domain = l2_errors(parameters, size, log)
                elapsed = time.perf_counter() - start
                domain = f"{domain:.6e}" if domain is not None else ""
                print(f"{size:>5} {error:>11.3e} {error / norm:>11.3e} {domain[:5] + domain[-4:]:>11} "
                      f"{paper.get(size, ''):>9} {elapsed:>9.1f}", flush=True)
                write_result(path, {"case": case, "N": size, "l2_error": f"{error:.6e}",
                                    "relative_error": f"{error / norm:.6e}", "rate": "",
                                    "paper_wsbm": paper.get(size, ""), "time_s": f"{elapsed:.2f}",
                                    "relative_error_domain": domain})


if __name__ == "__main__":
    main()
