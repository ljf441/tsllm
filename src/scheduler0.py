import math
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator

from collections.abc import Iterable

from preprocessor import load_and_preprocess, decoding, process_data
from qwen import load_qwen

import numpy as np

import matplotlib.pyplot as plt

from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from transformers import get_cosine_schedule_with_warmup
from preprocessor import get_dataset

import wandb
import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


import gc

import os
import glob
import re

torch.cuda.empty_cache()

np.random.seed(442)

#for matplotlib plots
SMALL_SIZE = 15+5
MEDIUM_SIZE = 20+5
BIGGER_SIZE = 25+5

plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=SMALL_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Define LoRA layers
class LoRALinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, r: int, alpha: int = None):
        super().__init__()
        assert isinstance(original_linear, nn.Linear)
        self.original_linear = original_linear
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha if alpha else r

        device = original_linear.weight.device
        self.A = nn.Parameter(torch.empty(r, in_dim, device=device))
        self.B = nn.Parameter(torch.zeros(out_dim, r, device=device))
        
        # Initialise A with He initialization
        nn.init.kaiming_normal_(self.A, nonlinearity="linear")

    def forward(self, x):
        base_out = self.original_linear(x)
        lora_out = (x @ self.A.T) @ self.B.T
        return base_out + lora_out * (self.alpha / self.r)
    
# Load the model and tokenizer
model, tokenizer = load_qwen()

# Modified tokenization with chunking
def process_sequences(texts, tokenizer, max_length=512, stride=256):
    all_input_ids = []
    for text in texts:
        # Apply Qwen's tokenization scheme to the text:
        encoding = tokenizer(text, return_tensors="pt", add_special_tokens=False, padding_side='left')
        seq_ids = encoding.input_ids[0]

        # Create sliding windows to further divide the data into chunks:
        for i in range(0, len(seq_ids), stride):
            chunk = seq_ids[i : i + max_length]
            if len(chunk) < max_length:
                chunk = torch.cat(
                    [
                        torch.full((max_length - len(chunk),), tokenizer.pad_token_id),
                        chunk,
                    ]
                )
            all_input_ids.append(chunk)
    return torch.stack(all_input_ids)

# Process the testing data into sequences of text as well as input IDs
def process_data(texts, tokenizer, points=80):
    """
    Process the data into sequences of text
    
    Args:
        texts: list of original strings
        tokenizer: tokenizer object
        points: number of points to give to the model
        
    Returns:
        np.array: texts
        torch.Tensor: given_input_ids
    """
    given_input_ids = []
    for text in texts:
        given_text = ';'.join([chunk for i, chunk in enumerate(text.split(';')) if i < points])
        encoding_given = tokenizer(given_text, return_tensors="pt", padding='max_length', padding_side='left', max_length=1200)
        given_input_ids.append(encoding_given.input_ids[0])
    return np.stack([text for text in texts]), torch.stack(given_input_ids)

def running_mse(prediction, actual):
    """
    Calculate the running mean squared error.

    Args:
        prediction: list of predicted values
        actual: list of actual values

    Returns:
        np.array: running mean squared error
    """
    mse = []
    for i in range(len(prediction)):
        mse.append(mean_squared_error(prediction[:i+1], actual[:i+1]))
    return np.array(mse)

def evaluate_model(model, val_loader, step):
    """
    Evaluate the model on the validation set.
    
    Args:
        model: model to evaluate
        val_loader: validation data loader
        step: current step

    Returns:
        float: average loss on the validation set
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(tqdm(val_loader, desc="val set")):
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
            
    # Calculate metrics
    num_batches = len(val_loader)
    avg_loss = total_loss / len(val_loader)
    # print(f'Loss on validation subset ({num_batches}/{len(val_loader)} batches) at step {step}: {avg_loss:.4f}')
    return avg_loss

def move_to_cpu(obj):
    """Recursively convert tensors to CPU NumPy arrays."""
    if isinstance(obj, torch.Tensor):
        # Move tensor to CPU and convert to NumPy
        return obj.detach().cpu().numpy()
    elif isinstance(obj, np.ndarray):
        # Already a NumPy array (no action needed)
        return obj
    elif isinstance(obj, dict):
        # Process dictionary values
        return {k: move_to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        # Process lists, tuples, etc.
        return type(obj)(move_to_cpu(v) for v in obj)
    else:
        # Return Python primitives (int, float, etc.) as-is
        return obj
    
# Define constants
lora_rank = 8
lora_alpha = 2*lora_rank
batch_size = 4
learning_rate = 0.0001
test_size = 0.2
max_steps = 400
max_ctx_length = 768
points = 80

schedulers = ['StepLR', 'CosineAnnealingLR', 'CosineScheduleWithWarmup']

schedule_choice = schedulers[0]

np.random.seed(442)

print(schedule_choice)

# Load the model and tokenizer
model, tokenizer = load_qwen()

# Process the data into sequences of text
train_texts, val_texts, test_texts = load_and_preprocess("lotka_volterra_data.h5", test_size=test_size)

# Defines the maximum context length for the model
train_input_ids = process_sequences(
    train_texts, tokenizer, max_ctx_length, stride=max_ctx_length // 2
)
val_input_ids = process_sequences(
    val_texts, tokenizer, max_ctx_length, stride=max_ctx_length
)
test_texts_all, test_input_ids_some = process_data(
    test_texts, tokenizer, points=points
)

# Create data loaders

train_dataset = TensorDataset(train_input_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_input_ids)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_input_ids_some)
test_loader = DataLoader(test_dataset, shuffle=False)

# Dictionary to store results
grid_results = {}

# print(f"\n{'='*50}")
# print(f"Training with scheduler={schedule_choice}")
# print(f"{'='*50}\n")

# Actually apply LoRA to the model:
for layer in model.model.layers:
    layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, r=lora_rank, alpha = lora_alpha)
    layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, r=lora_rank, alpha = lora_alpha)
# ^These are the parts that will actually be trained!

for name, param in model.named_parameters():
    if "A" in name or "B" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Freeze all layers except the LoRA layers
for name, param in model.named_parameters():
    if "A" in name or "B" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Create optimizer with current learning rate
optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), 
    lr=learning_rate, 
)

# Create scheduler
if schedule_choice == 'StepLR':
    step_size = 100
    gamma = 0.1
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
elif schedule_choice == 'CosineAnnealingLR':
    T_max = max_steps
    scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
elif schedule_choice == 'CosineScheduleWithWarmup':
    warmup_steps = int(0.075*max_steps)
    num_training_steps = max_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

#Prepare with accelerator
accelerator = Accelerator()
model, optimizer, scheduler, train_loader_local, val_loader_local, test_loader_local = accelerator.prepare(
    model, optimizer, scheduler, train_loader, val_loader, test_loader
)

print(schedule_choice)

# Train the model (shortened training for grid search)
steps = 0
train_losses = []
val_losses = []
early_stop_steps = min(max_steps, 500)  # Reduce training for grid search

while steps < early_stop_steps:
    progress_bar = tqdm(train_loader_local, desc=f"Steps {steps}")
    for (batch,) in progress_bar:
        model.train()
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        train_losses.append([loss.item(), steps])
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        
        if steps % 50 == 0:
            avg_loss = evaluate_model(model, val_loader_local, steps)
            val_losses.append([avg_loss, steps])
            model.train()
            
        steps += 1
        progress_bar.set_postfix(loss=loss.item())

        del loss
        del outputs
        del batch
        
        if steps >= early_stop_steps:
            break

# Final evaluation
final_val_loss = evaluate_model(model, val_loader_local, steps)

if schedule_choice == 'StepLR':
    grid_results[(schedule_choice)]["step_size"] = step_size
    grid_results[(schedule_choice)]["gamma"] = gamma
elif schedule_choice == 'CosineAnnealingLR':
    grid_results[(schedule_choice)]["T_max"] = T_max
elif schedule_choice == 'CosineScheduleWithWarmup':
    grid_results[(schedule_choice)]["warmup_steps"] = warmup_steps
    grid_results[(schedule_choice)]["num_training_steps"] = num_training_steps

# Test the model
model.eval()
with torch.no_grad():    
    for (batch,) in tqdm(test_loader_local):
        outputs = model.generate(batch, attention_mask = torch.ones_like(batch), max_new_tokens=max_ctx_length*2)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_decoded = decoding(prediction)
        break

test_decoded = decoding(test_texts_all[0])

# Clean up
del model
del tokenizer
del optimizer
del train_loader_local
del val_loader_local
del test_loader_local
del accelerator
del scheduler

# Store results
grid_results[(schedule_choice)] = {
    "final_val_loss": final_val_loss,
    "train_losses": train_losses,
    "val_losses": val_losses,
    "test_decoded": test_decoded,
    "prediction_decoded": prediction_decoded,
}

grid_result = move_to_cpu(grid_results)

# Save results
torch.save(grid_results, f"../results/more_grid_results_{schedule_choice}.pt")
# joblib.dump(grid_results, f"../results/grid_results_{schedule_choice}.gz", compress=3)

del train_losses
del val_losses
torch.cuda.empty_cache()