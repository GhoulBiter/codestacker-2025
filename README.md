# CityX Crime Analysis

osme description

## Author Details

## Table of Contents

## Requirements

## Installation

Below is a concise, beginner-friendly README.md to help set up Miniconda, pipx, and Poetry with reproducibility across different OSes:

---

```markdown
# Project Setup: Conda, pipx & Poetry

This guide helps you install Miniconda, pipx, and Poetry for reproducible Python project environments. It covers Linux (Ubuntu), macOS, and Windows (using WSL or CMD/PowerShell).

## 1. Install Miniconda

- **Linux/macOS:**
  1. Download the installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
  2. Run the installer:
     ```bash
     bash Miniconda3-latest-Linux-x86_64.sh   # or macOS equivalent
     ```
  3. Follow the prompts and restart your terminal.

- **Windows:**
  1. Download the Windows installer from [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
  2. Run the installer and follow the setup instructions.

## 2. Install pipx via conda

Create a dedicated environment for CLI tools (optional but recommended):

```bash
conda create -n cli-tools -c conda-forge pipx
conda activate cli-tools
```

Ensure pipx's bin directory is in your PATH:

- **Linux/macOS (bash/zsh):**  
  Add to `~/.bashrc` or `~/.zshrc`:

  ```sh
  export PATH="$PATH:$HOME/.local/bin"
  ```

- **Windows:**  
  pipx installs to `%USERPROFILE%\.local\bin`. Add this path to your system/user PATH if not automatically done.

Run:

```bash
pipx ensurepath
```

Restart your terminal if needed.

## 3. Install Poetry using pipx

With pipx installed (and optionally, the `cli-tools` environment active), run:

```bash
pipx install poetry
```

Verify with:

```bash
poetry --version
```

## 4. Start a New Project

There are two options:

### Option A: Let Poetry Manage Its Virtual Environment (Default)

1. Create a new project:

   ```bash
   poetry new my-project
   cd my-project
   ```

2. Add dependencies:

   ```bash
   poetry add streamlit pandas scikit-learn scipy nltk numpy@1.26.4 seaborn statsmodels transformers sentence-transformers fastapi pdfplumber plotly geopandas folium dash keplergl shap lime xgboost lightgbm catboost h3 streamlit-folium streamlit-plotly-events streamlit-lottie dask ray
   ```

3. Work within the Poetry shell:

   ```bash
   poetry shell
   ```

### Option B: Use an Existing Conda Environment

1. Create and activate your conda environment:

    ```bash
   conda create -n my-project-env python=3.11
   conda activate my-project-env
   ```

2. In your project directory, tell Poetry to use the current environment:

    ```bash
   poetry config virtualenvs.create false --local
   ```

3. Initialize and add dependencies:

    ```bash
   poetry init
   poetry add streamlit pandas scikit-learn scipy nltk numpy seaborn statsmodels transformers fastapi
   ```

## Summary

- **Miniconda**: Manage your Python installations.
- **pipx**: Install and manage CLI tools (like Poetry) isolated from your projects.
- **Poetry**: Create reproducible project environments with dependency management.

Use the method (Option A or B) that best fits your workflow. Happy coding!
