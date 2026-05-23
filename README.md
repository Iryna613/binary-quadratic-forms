# Conway Topograph Analyzer

Interactive Streamlit application for analyzing integral binary quadratic forms

\[
Q(x,y)=Ax^2+Bxy+Cy^2
\]

with Conway topograph visualization, discriminant and type analysis, Gauss reduction, topographic reduction, local consistency checks, representation search, and performance benchmarking.

## Features

The application allows you to:

- enter coefficients `A`, `B`, `C`,
- compute the discriminant `D = B^2 - 4AC`,
- classify the form (`positive definite`, `negative definite`, `indefinite`, `degenerate`),
- check primitiveness (`gcd(A, B, C) = 1`),
- generate a bounded Conway topograph fragment (configurable compute depth),
- display region values and optional lax vectors,
- detect `well` for positive definite forms,
- highlight `river` for indefinite forms,
- run Gauss reduction (positive definite forms),
- compare Gauss and topographic reduction outputs,
- validate local invariants:
  - primitive regions,
  - unimodular edges,
  - Conway diamond rule,
- search integer representations of `Q(x,y)=N` within a bound,
- export:
  - topograph and benchmark plots to HTML/PNG,
  - analysis output to JSON,
  - benchmark tables to CSV.

## Technologies

- Python 3.10+
- Streamlit (UI)
- Plotly + Kaleido (interactive plots and PNG export)
- Pandas (tables, benchmark outputs)
- Custom modules in `topograph_analyzer/` for forms, reduction, and topograph logic

## Current Project Structure

```text
binary-quadratic-forms/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|-- streamlit_app.py
|-- outputs/
`-- topograph_analyzer/
    |-- __init__.py
    |-- quadratic_form.py
    |-- gauss.py
    |-- topograph.py
    `-- classic_view.py
```

Note: this repository currently does not contain `cli.py`, `__main__.py`, `examples/`, or `tests/`.

## Requirements

Before running the app, make sure you have:

- Python 3.10 or newer,
- `pip`,
- dependencies from `requirements.txt`.

Using a virtual environment is recommended.

## Installation

Linux / macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running the Application

```bash
streamlit run streamlit_app.py
```

After startup, open:

`http://localhost:8501`

## How to Use

1. Select a predefined example or enter custom `A, B, C`.
2. Set compute depth and display depth.
3. Enable optional display settings (lax vectors, helper vertices, sign coloring).
4. Optionally enable representation search `Q(x,y)=N` with a bound.
5. Review results across tabs:
   - Topograph
   - Reduction
   - Checks
   - Regions
   - JSON export
   - Benchmark

## UI Outputs

- **Topograph:** fixed Conway-style template with computed values; river edges are highlighted for indefinite forms.
- **Reduction:** Gauss reduction, topographic candidate, and direct comparison.
- **Checks:** primitive-region, unimodular-edge, and diamond-rule validation.
- **Regions:** generated region table and optional representation solutions.
- **JSON export:** downloadable structured analysis result.
- **Benchmark:** timing comparison of reduction methods and depth scaling.

## Example

For

\[
Q(x,y)=4x^2+7xy+9y^2
\]

the app computes discriminant, type, reduction data, topograph fragment, and benchmark-ready outputs.

## Troubleshooting

Common issues:

- virtual environment not activated,
- dependencies not installed,
- Streamlit not installed in active environment.

If startup fails, reinstall dependencies and verify the command:

```bash
streamlit run streamlit_app.py
```

## Testing

`pytest` is listed in dependencies, but this repository currently has no committed test suite.

## Summary

The project provides a practical environment for combining algebraic analysis of binary quadratic forms with Conway topograph visualization and reduction experiments.
