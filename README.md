# MRMU Validation Suite

![PyPI version](https://img.shields.io/pypi/v/validation-suite.svg)

A unified framework for consistent validation across multiple projects and use cases.

* [GitHub](https://github.com/ccsarmientot/validation-suite/) | [PyPI](https://pypi.org/project/validation-suite/) | [Documentation](https://ccsarmientot.github.io/validation-suite/)
* Created by [Cristian C. Sarmiento](http://linkedin.com/in/ccsarmientot) | GitHub [@ccsarmientot](https://github.com/ccsarmientot) | PyPI [@ccsarmientot](https://pypi.org/user/ccsarmientot/)
* MIT License

## Features

* Instalación y configuración
* `compare_dataframes` — Benchmarking / Dry Run vs Model Owner
* `vif_check` — Test de multicolinealidad (supuestos de regresión)
* `psi_check` — Population Stability Index (estabilidad de inputs)
* `csi_check` — Characteristic Stability Index (estabilidad por segmento)
* Exportar evidencia a Excel (trazabilidad para auditoría)
* Flujo completo de validación

## Documentation

Documentation is built with [Zensical](https://zensical.org/) and deployed to GitHub Pages.

* **Live site:** https://ccsarmientot.github.io/validation-suite/
* **Preview locally:** `just docs-serve` (serves at http://localhost:8000)
* **Build:** `just docs-build`

API documentation is auto-generated from docstrings using [mkdocstrings](https://mkdocstrings.github.io/).

Docs deploy automatically on push to `main` via GitHub Actions. To enable this, go to your repo's Settings > Pages and set the source to **GitHub Actions**.

## Development

To set up for local development:

```bash
# Clone your fork
git clone git@github.com:ccsarmientot/validation-suite.git
cd validation-suite

# Install in editable mode with live updates
uv tool install --editable .
```

This installs the CLI globally but with live updates - any changes you make to the source code are immediately available when you run `validation_suite`.

Run tests:

```bash
uv run pytest
```

Run quality checks (format, lint, type check, test):

```bash
just qa
```

## Author

MRMU Validation Suite was created in 2026 by Cristian C. Sarmiento.

Built with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [audreyfeldroy/cookiecutter-pypackage](https://github.com/audreyfeldroy/cookiecutter-pypackage) project template.
