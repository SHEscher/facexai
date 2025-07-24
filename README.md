# XAI methods for face recognition models

## Hands-on notebook

We use a [`marimo`](https://marimo.io) notebook.
`marimo` is a recently published alternative to `Jupyter` notebook –
its focus is on reproducibility and interactivity.

We run the [`marimo`](https://marimo.io) notebook using the package manager [`uv`](https://docs.astral.sh/uv/).
This is recommended in order to run the notebook in a sandboxed environment.
That is, all requirements are listed in the notebook script itself.

Check the section [Requirements](#requirements) for information on how to install `uv`.

### Run the `marimo` notebook

```shell
uvx marimo edit --sandbox facexai.py
```

### Add packages

```shell
uv add --script=facexai.py [package]
```

Or run the notebook and import the respective package.
If the package is not installed yet,
you can install it via `marimo` within the notebook.

## Data

Required data can be downloaded via the notebook.

You can manually download the data also from: https://keeper.mpdl.mpg.de/f/4efc6def1acd4de8a84d/

Unzip the downloaded file and move its content to `./data/`.


## Requirements

Install the `Python` package manager `uv`

### Install `uv` on macOS & Linux

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install `uv` on Windows

```shell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

For other installation methods, see the `uv` [docs](https://docs.astral.sh/uv/getting-started/installation/).

## Housekeeping

Some recommendations after being done:

### Clean up with `uv`

The (great) Python package manager `uv` caches packages to reuse them across projects.
If you do not intend to use `uv` further, consider deleting the cache of `uv`.

Run the following in your terminal:

```shell
uv cache clean
```

### Clean up the results

You only need the files `facexai.py`, `README.md`, and the `./data/` folder to reproduce the pipeline.
Consider deleting the `./results/` folder to save some disk space after running the script.

## Contact

Simon M. Hofmann – simon.hofmann[ät]cbs.mpg.de
