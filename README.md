# A/B Testing Examples
This marimo notebook demonstrates some of the benefits to using Bayesian methods for A/B testing.

Some benefits include:
1. Using informed priors to improve estimates.
2. Outputting distributions rather than just point-wise estimates.
3. Ability to use smaller sample sizes.
4. Easier interpretability regarding statistical significance.

## Set up

### uv
The Python package manager `uv` can make your life a lot easier.

Install it [here](https://docs.astral.sh/uv/getting-started/installation/).

### Virtual Environment
Set up your Python virtual environment:

```bash
uv venv
```

### Marimo
Install marimo:

```bash
uv pip install marimo
```

## Usage
Run the notebook in sandbox mode:

Note: all of the necessary dependencies are installed on start up because of the --sandbox flag.

```bash
uv run marimo edit --sandbox notebook.py
```