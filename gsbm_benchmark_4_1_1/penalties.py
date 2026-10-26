"""
Penalty study of the standard Taylor shift, without the shifted subscale, at
p = 3: every case on N x N knot spans for every combination of the Nitsche
factor chi and the ghost penalty factor kappa, with the other settings of its
ProjectParameters json and of config/MaterialParameters_standard_p3.json.
The materials of every combination are written to out/penalties/, with the
ghost penalty kappa / ((p!)^2 (2p + 1)). Writes the L2 error of Eq. (48), the
relative one, ||u - u_h|| / ||u||, and the rate to out/penalties_p3.csv after
every run, replacing only the cases, factors and meshes run, and appends the
Kratos log to out/penalties_p3.log. All standard cases by default:

    python3 penalties.py [config/ProjectParameters_<case>_standard.json ...] [--chi 2 20 200 2000]
        [--kappa 0.05 0.5 5 50] [--sizes 40 80 160 320]

figures.py plots the results.
"""

import argparse
import csv
import fcntl
import json
import os
import time
from math import factorial
from pathlib import Path

import numpy as np

import KratosMultiphysics as KM

from convergence import l2_errors, set_degree, set_materials

DEGREE = 3
CHI = (2., 20., 200., 2000.)
KAPPA = (.05, .5, 5., 50.)
SIZES = (40, 80, 160, 320)
MATERIALS = Path("config/MaterialParameters_standard_p3.json")
COLUMNS = ("case", "chi", "kappa", "N", "l2_error", "relative_error", "rate", "time_s")


def write_materials(chi, kappa):
    """The materials of MATERIALS with the factors chi and kappa, in out/penalties/."""
    materials = json.loads(MATERIALS.read_text())
    variables = materials["properties"][0]["Material"]["Variables"]
    variables["PENALTY_FACTOR"] = chi
    variables["GHOST_PENALTY_FACTOR"] = kappa / (factorial(DEGREE)**2 * (2 * DEGREE + 1))
    path = Path(f"out/penalties/MaterialParameters_chi{chi:g}_kappa{kappa:g}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(materials, indent=4) + "\n")
    return path


def write_result(path, result):
    """Merges a result, keyed by case, factors and mesh, into a CSV file: it
    replaces the row of the same key and keeps the others, and the rates
    between successive meshes of every case and factors are recomputed. Locked,
    for runs side by side."""
    def key(row):
        return row["case"], float(row["chi"]), float(row["kappa"])

    with open(path, "a") as lock:  # the file itself, rewritten in place
        fcntl.flock(lock, fcntl.LOCK_EX)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        rows = [r for r in rows if (key(r), int(r["N"])) != (key(result), int(result["N"]))] + [result]
        lines = [",".join(COLUMNS)]
        for group in dict.fromkeys(key(r) for r in rows):
            previous = None
            for row in sorted((r for r in rows if key(r) == group), key=lambda r: int(r["N"])):
                size, error = int(row["N"]), float(row["l2_error"])
                row["rate"] = f"{np.log(previous[1] / error) / np.log(size / previous[0]):.3f}" if previous else "nan"
                previous = (size, error)
                lines.append(",".join(str(row[column]) for column in COLUMNS))
        path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", type=Path, help="ProjectParameters json files (all in config/)")
    parser.add_argument("--chi", nargs="+", type=float, default=CHI, help="Nitsche penalty factors")
    parser.add_argument("--kappa", nargs="+", type=float, default=KAPPA, help="ghost penalty factors")
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES, help="knot spans per direction")
    arguments = parser.parse_args()

    # The json files give the skins, materials and outputs relative to this folder
    here = Path(__file__).resolve().parent
    files = [file.resolve() for file in arguments.files] or sorted(here.glob("config/ProjectParameters_*_standard.json"))
    os.chdir(here)
    log, path = Path("out/penalties_p3.log"), Path("out/penalties_p3.csv")

    for file in files:
        parameters = KM.Parameters(file.read_text())
        set_degree(parameters, DEGREE)
        case = parameters["problem_data"]["problem_name"].GetString()
        for chi in arguments.chi:
            for kappa in arguments.kappa:
                set_materials(parameters, str(write_materials(chi, kappa)), DEGREE)
                print(f"\n{case}, p = {DEGREE}, chi = {chi:g}, kappa = {kappa:g}")
                print(f"{'N':>5} {'L2 error':>11} {'relative':>11} {'time [s]':>9}")
                for size in arguments.sizes:
                    start = time.perf_counter()
                    error, norm = l2_errors(parameters, size, log)
                    elapsed = time.perf_counter() - start
                    print(f"{size:>5} {error:>11.3e} {error / norm:>11.3e} {elapsed:>9.1f}", flush=True)
                    write_result(path, {"case": case, "chi": f"{chi:g}", "kappa": f"{kappa:g}", "N": size,
                                        "l2_error": f"{error:.6e}", "relative_error": f"{error / norm:.6e}",
                                        "rate": "", "time_s": f"{elapsed:.2f}"})


if __name__ == "__main__":
    main()
