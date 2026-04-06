## Install UV using powershell:
UV must be installed using powershell
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Restart bash an verify uv version using
```bash
uv --version
$env:UV_LINK_MODE = "copy"
```

## Install cookie-cutter ussing uv (avoid Onedrive soft links)
This code is used to avoid error when working with Onedrive as store manager
```bash
uv tool install cookiecutter --link-mode copy 
```

This code creates a new repo inside the folder you choose
```bash
cookiecutter gh:audreyfeldroy/cookiecutter-pypackage
```

Note: Create with diff names in import and repo name


## Create venv with uv
This code creates a new repo inside the folder you choose
```bash
## Create with
uv venv

## Pin stable python version
uv python pin 3.12
```

## Test
Recreate steps for testing
```bash
## Add dev libraries to env
uv sync --extra dev --link-mode copy

## Run pytest then
uv run pytest tests/test_compare.py -v

## Run coverage tests

# optional if just not installed
uv tool install rust-just --link-mode copy

# Run coverage python version tests
just coverage

```


## Publishing proyect

To publish is required create an account using pypi and testpypi. 

When created:

### For testpypi:

Recommended to deploy here first

* Account Settings -> API Tokens -> Add API Token

* name: uv-token  -> scope: "Entire account".

* Copy Token


With the token created, then build the distribution package
```bash
uv build
```

And then, publish it ussing the gotten token (Twine not needed):
```bash
uv publish --publish-url https://test.pypi.org/legacy/ --token <YOUR_TOKEN>
```

To intall the published test library use:
```bash
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ validation-suite
```

[Optional] Complement justfile:
```bash
# Publicar versión de prueba en TestPyPI
publish-test:
    uv build
    uv publish --publish-url https://test.pypi.org/legacy/
```

### For pypi (production):

* Account Settings -> API Tokens -> Add API Token

* name: uv-prod-token  -> scope: "Entire account".

* Copy Token


Delete dist/ if exists
```bash
rm -rf dist/
```

With the token created, then build the distribution package
```bash
uv build
```

And then, publish it ussing the gotten token (Twine not needed):
```bash
uv publish --token <YOUR_PROD_TOKEN>
```

To intall the published library:
```bash
uv pip install validation-suite
```

[Optional] Complement justfile:
```bash
# publish to pypi
publish:
    uv build
    uv publish
```

## Versioning

Install bump-my-version
```bash
uv tool install bump-my-version --link-mode copy
```

Add this in the final secction of pyproject.toml
```bash
[tool.bumpversion]
current_version = "0.1.0"
parse = "(?P<major>\\d+)\\.(?P<minor>\\d+)\\.(?P<patch>\\d+)"
serialize = ["{major}.{minor}.{patch}"]
search = "{current_version}"
replace = "{new_version}"

[[tool.bumpversion.files]]
filename = "pyproject.toml"

[[tool.bumpversion.files]]
filename = "src/validation_suite/__init__.py"
```

Add to justfile
```bash
set shell := ["powershell.exe", "-Command"]

# Sube la versión mínima (0.1.0 -> 0.1.1) - Errores pequeños
patch:
    uvx bump-my-version bump patch

# Sube la versión media (0.1.0 -> 0.2.0) - Nueva funcionalidad
minor:
    uvx bump-my-version bump minor

# Sube la versión mayor (0.1.0 -> 1.0.0) - Cambios que rompen todo
major:
    uvx bump-my-version bump major
```

## Documentation
Make sure to have in pyproject.toml the next lines:
```bash
[dependency-groups]
docs = [
    "zensical",
    "sphinx",
    "mkdocs",
    "mkdocstrings[python]",

]
```

Then is easy, just use:
```bash
## Just version
just docs-build

## Direct code
uv run --group docs zensical build --clean
```

To see live
```bash
## Just version
just docs-serve
```