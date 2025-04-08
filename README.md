# hpc branch
This branch is designed to be used on CSD3, such that all other extraneous folders have been removed.

# how to use
Create an Anaconda `conda` environment:
```
conda create -n lora matplotlib numpy pytorch torchvision torchmetrics -c pytorch 
```
Then activate it:
```
conda activate lora
```
Then install all necessary packages from the `req.txt` file:
```
pip install -r req.txt
```
Then, using Slurm and the provided `.wilkes3` submission script, jobs can be submitted to CSD3. It is recommended to use one node only as training is not particularly intensive.

Expected wall time:
- `gridsearch.py`: ~45 minutes.
- `lora_default.py`: ~30 minutes.
- `lora_run.py`: ~3 hours.
