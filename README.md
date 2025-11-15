# Revised Simplex Method for Solving Linear Programming Problems

This repository contains a scientific study and LaTeX report on the **Revised Simplex Method**, an efficient variant of the classical Simplex Algorithm for solving **Linear Programming Problems (LPP)**. It includes a complete report, experimental results, and comparison with SciPy’s HiGHS solver.

---

## Project Structure

```

revised-simplex-report/
│
├── main.tex                         # Main LaTeX file
├── docs/                          
│   └── revised_simplex_report.pdf   # Compiled PDF report
├── bib/
│   └── references.bib               # BibTeX references
├── code/
│   ├── revised_simplex.py           # Python implementation of the Revised Simplex
│   └── run_experiments.py           # Benchmark script for test problems
├── results/                         # Per-run JSON logs and aggregated CSV
│   ├── example1_revised_run*.json
│   ├── example1_scipy_run.json
│   ├── example2_phaseI_revised_run*.json
│   ├── example2_phaseI_scipy_run.json
│   └── summary.csv
├── LICENSE                           # CC BY-SA 4.0 license
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies
└── .gitignore                         # Git ignore file

````

---

## Report Overview

The LaTeX report (`main.tex`) is structured as follows:

| Chapter             | Description                                                                 |
| ------------------- | --------------------------------------------------------------------------- |
| Motivation          | Explains the need for the Revised Simplex Method and its advantages.        |
| Theory              | Mathematical formulation, basis matrices, reduced costs, and optimality.    |
| Algorithm           | Pseudocode for Phase I and Phase II, pivot selection, feasibility checks.   |
| Implementation      | Notes on Python implementation, numerical safeguards, and data structures. |
| Examples            | Worked examples for regular, two-phase, and special-case LP problems.       |
| Comparison          | Performance and accuracy comparison with SciPy’s HiGHS solver.             |
| Conclusion          | Summary of findings and suggested extensions.                               |

---

## Requirements

### LaTeX

To compile the report locally, ensure the following packages are installed:

* `amsmath`, `amssymb`, `graphicx`, `geometry`, `algorithm`, `algpseudocode`, `booktabs`, `hyperref`

### Python

* Python 3.9 or higher
* `numpy`
* `scipy`
* `pandas` (optional, for data handling)

Install dependencies using:

```bash
pip install -r requirements.txt
````

---

## Compiling the Report

### Local Compilation

```bash
pdflatex -output-directory=docs -jobname=revised_simplex_report main.tex
biber docs/revised_simplex_report
pdflatex -output-directory=docs -jobname=revised_simplex_report main.tex
pdflatex -output-directory=docs -jobname=revised_simplex_report main.tex
```

The resulting PDF will be located at:

```
docs/revised_simplex_report.pdf
```

### Overleaf

Upload all `.tex` files, the `bib/` folder, and `code/` folder. Open `main.tex` and recompile to produce the PDF.

---

## Results

The `results/` folder contains:

* JSON logs for each run (`example1_revised_run*.json`, `example2_phaseI_revised_run*.json`, etc.)
* Aggregated summary CSV (`summary.csv`) used for tables and figures in the report

The report includes a detailed comparison with SciPy’s HiGHS solver in terms of:

* Objective value correctness
* Feasibility residuals
* Iterations/pivots and basis diagnostics
* Timing and performance

---

## Authors

* **Amr Yasser Anwar** — Zewail City, Department of CSAI
* **Omar Hazem Ahmed** — Zewail City, Department of CSAI

---

## License

This project is licensed under the [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).
