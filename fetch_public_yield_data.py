"""Download public reaction-yield datasets used by this project."""

from __future__ import annotations

import argparse
import os
import urllib.request


URLS = {
    "bh_reactions_csv": "https://raw.githubusercontent.com/schwallergroup/ai4chem_course/main/notebooks/10%20-%20Bayesian%20optimization/bh-reactions.csv",
    "dreher_doyle_xlsx": "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx",
    "suzuki_xlsx": "https://raw.githubusercontent.com/rxn4chemistry/rxn_yields/master/data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx",
}


def download(url: str, path: str) -> None:
    urllib.request.urlretrieve(url, path)  # noqa: S310


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public HTE reaction-yield datasets.")
    parser.add_argument("--output-dir", type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for name, url in URLS.items():
        ext = ".csv" if name.endswith("csv") else ".xlsx"
        target = os.path.join(args.output_dir, f"{name}{ext}")
        download(url, target)
        print(f"Downloaded {name} -> {target}")


if __name__ == "__main__":
    main()
