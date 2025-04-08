# tsllm
Training a LoRA-adapted Qwen2.5-0.5B-Instruct LLM to be able to predict Lotka-Volterra time series data.

# how to run
It is recommended to run this project within a virtual environment. To run the `.py` files using CSD3, it is recommended to follow the instructions for creating the `conda` environment and using it when submitting a job.

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
Create the conda environment:
```
conda env create -f environment.yml
```
or:
```
conda create -n lora matplotlib numpy pytorch torchvision torchmetrics -c pytorch
```
Then activate:
```
conda activate lora
```
Then install all necessary packages from the `requirements.txt` file:
```
pip install -r requirements.txt
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
2. `.py` files:
    - `qwen.py` and `lora_skeleton.py`, which were provided.
    - `preprocessor.py` covering task 2a of the coursework.
    - `flops.py` covering tasks 1 and 2c of the coursework.
    - `lora_default.ipynb` covering task 3a of the coursework.
    - `gridsearch.py` covering task 3b of the coursework.
    - `lora_run.py` covering task 3c of the coursework.

This project contains within the `models` folder:
1. `.pt` file containing the LoRA weights
2. `.pth` file containing the optimizer state

This project contains within the `results` folder:
1. `best_overall_params.pt`, containing a dictionary of the overall parameters.
2. `grid_results*.pt` files, containing the training and validation losses for each hyperparameter search.
3. `lora_run*.pt` files, containing the training and validation losses for the default and final training runs.


# where to find the report
The report is located in the `report` folder and is named `report.pdf`.

# use of auto-generation tools

Auto-generation tools were used as follows:
- Parsing error messages throughout the project.
- Creating a `move_to_cpu` function to move tensors on the GPU to the CPU.
- Helping to create a `decoding` function to strip out LLM-injected errors in the output when transforming to time series arrays.
- Assistance in formatting the report in $\LaTeX$, specifically with tables and referencing.

Auto-generation tools were not used elsewhere, for code generation, writing, or otherwise.

# acknowledgements

The $\LaTeX$ bibliography style was taken from https://github.com/gbhutani/vancouver_authoryear_bibstyle.git
