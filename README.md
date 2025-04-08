# hpc branch
This branch is designed to be used on CSD3, such that all other extraneous folders have been removed.

# how to use
Login onto CSD3. Git clone this repository.
```
git clone https://gitlab.developers.cam.ac.uk/phy/data-intensive-science-mphil/assessments/m2_coursework/lj441.git
```
Then, checkout to the `hpc` branch:
```
git checkout hpc
```
Create an Anaconda `conda` environment:
```
conda create -n lora matplotlib numpy pytorch torchvision torchmetrics -c pytorch 
```
Then activate it:
```
conda activate lora
```
Then install all necessary packages from the `requirements.txt` file:
```
pip install -r requirements.txt
```
Then, using Slurm and the CSD3 provided `.wilkes3` submission script (not found in this repository), jobs can be submitted to CSD3. It is recommended to use one node only as training is not particularly intensive.

Expected wall time:
- `gridsearch.py`: ~45 minutes.
- `lora_default.py`: ~30 minutes.
- `lora_run.py`: ~3 hours.
