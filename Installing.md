### Install UV using powershell:
UV must be installed using powershell
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
Restart bash an verify uv version using
```bash
uv --version
$env:UV_LINK_MODE = "copy"
```

### Install cookie-cutter ussing uv (avoid Onedrive soft links)
This code is used to avoid error when working with Onedrive as store manager
```bash
uv tool install cookiecutter --link-mode copy 
```

This code creates a new repo inside the folder you choose
```bash
cookiecutter gh:audreyfeldroy/cookiecutter-pypackage
```

Note: Create with diff names in import and repo name


### Create venv with uv
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


