# tsllm
Training a Qwen2 0.5 B LLM to be able to predict Lotka-Volterra time series data.

# how to run
It is recommended to run this project within a virtual environment. To run the `.py` files using CSD3, it is recommended to follow the instructions for the `conda` environment.

Two requirements files are provided, to use with either Python's `venv` or Anaconda's `conda`.

## Python
First, create a virtual environment:
```
python3 -m venv lora
```
Then activate:
```
source lora/bin/activate
```
Then, install the necessary modules from the `requirements.txt` file after activating the environment.
```
pip install -r requirements.txt
```

## Anaconda
Create the conda environment using `req.txt`:
```
conda create -n lora --file req.txt
```
Then activate:
```
conda activate lora
```

# within
This repository contains in the `src` folder:
1. Jupyter Notebooks:
    - `preprocessor.ipynb` covering task 2a of the coursework.
    - `evaluation.ipynb` covering task 2b of the coursework.
    - `flops_calculator.ipynb` covering tasks 1 and 2c of the coursework.
    - `lora_default.ipynb` covering task 3a of the coursework.
    - `grid_search.ipynb` covering task 3b of the coursework.
    - `lora_final.ipynb` covering task 3c of the coursework.
    - ...
2. `.py` files:
    - `qwen.py` and `lora_skeleton.py`, which were provided.
    - `preprocessor.py` covering task 2a of the coursework.
    - `flops.py` covering tasks 1 and 2c of the coursework.
    - `lora_default.ipynb` covering task 3a of the coursework.
    - `gridsearch.py` covering task 3b of the coursework.
    - `lora_run.py` covering task 3c of the coursework.


# where to find the report
The report is located in the `report` folder and is named `report.pdf`.

# use of auto-generation tools

Auto-generation tools were used as follows:
- To help setup `Tensorflow` on a WSL2 environment to be able to use the GPU.
- Parsing error messages throughout the project.
- Assistance in formatting the report in $\LaTeX$, specifically with tables and referencing.

Auto-generation tools were not used elsewhere, for code generation, writing, or otherwise.
