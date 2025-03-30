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

lora_rank = 8
lora_alpha = 2*lora_rank
batch_size = 4
learning_rate = 1e-4
test_size = 0.2
max_steps = 15000
max_ctx_length = 768
points = 80
T_max = max_steps

run_name = "CSD3_15k_seed"

np.random.seed(442)
# wandb.init(project="lora_qwen", config={
#     "learning_rate": learning_rate,
#     "batch_size": batch_size,
#     "lora_rank": lora_rank,
#     "max_steps": max_steps,
#     "max_ctx_length": max_ctx_length,
#     "test_and_val_size": test_size,
#     "T_max": T_max
# })

def running_mse(prediction, actual):
    mse = []
    for i in range(len(prediction)):
        mse.append(mean_squared_error(prediction[:i+1], actual[:i+1]))
    return mse

def evaluate_model(model, val_loader, step, max_batches=None):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_idx, (batch,) in enumerate(tqdm(val_loader, desc="val set")):
            # Exit loop after processing max_batches
            if max_batches is not None and batch_idx >= max_batches:
                break
            outputs = model(batch, labels=batch)
            loss = outputs.loss
            total_loss += loss.item()
            
    
    # Calculate metrics - divide by actual number of batches processed
    num_batches = min(len(val_loader), max_batches) if max_batches is not None else len(val_loader)
    avg_loss = total_loss / num_batches

    print(f'Loss on validation subset ({num_batches}/{len(val_loader)} batches) at step {step}: {avg_loss:.4f}')
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


model, tokenizer = load_qwen()
tokenizer.padding_side = 'left'

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


# Process the data into sequences of text
train_texts, val_texts, test_texts = load_and_preprocess("lotka_volterra_data.h5", test_size=test_size)
# ^Each of these is a `list[str]` representing contiguous parts of the time series,
#  in text form (using the LLMTIME scheme).

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


def process_data(texts, tokenizer, points=80):
    given_input_ids = []
    for text in texts:
        given_text = ';'.join([chunk for i, chunk in enumerate(text.split(';')) if i < points])
        encoding_given = tokenizer(given_text, return_tensors="pt", padding='max_length', padding_side='left', max_length=1200)
        given_input_ids.append(encoding_given.input_ids[0])
    return np.stack([text for text in texts]), torch.stack(given_input_ids)

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

optimizer = torch.optim.Adam(
    (p for p in model.parameters() if p.requires_grad), lr=learning_rate
)

# scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
scheduler = CosineAnnealingLR(optimizer, T_max=max_steps)
# scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps)

train_dataset = TensorDataset(train_input_ids)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(val_input_ids)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(test_input_ids_some)
test_loader = DataLoader(test_dataset, shuffle=False)

# Prepare components with Accelerator
accelerator = Accelerator()
model, optimizer, train_loader, scheduler, val_loader = accelerator.prepare(model, optimizer, 
                                                                                         train_loader, 
                                                                                         scheduler, 
                                                                                         val_loader
                                                                                         )

# Train the model
steps = 0
train_losses = []
val_losses = []
while steps < max_steps:
    progress_bar = tqdm(train_loader, desc=f"Steps {steps}")
    for (batch,) in progress_bar:
        model.train()
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        train_losses.append([loss.item(), steps])
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        steps += 1

        progress_bar.set_postfix(loss=loss.item())

        # wandb.log({
        #     "train_loss": loss.item(),
        #     "steps": steps,
        #     "lr": optimizer.param_groups[0]['lr'],
        #     })

        if (steps % 50) == 0:
            avg_loss = evaluate_model(model, val_loader, steps)
            val_losses.append([avg_loss, steps])
            model.eval()

            # wandb.log({
            # "val_loss": avg_loss,
            # "steps": steps,
            # })

        if steps > max_steps:
            break    

# Unwrap model from accelerator
unwrapped_model = accelerator.unwrap_model(model)

# Create a dictionary to store LoRA weights
lora_state_dict = {}

# Extract LoRA weights from each layer
for i, layer in enumerate(unwrapped_model.model.layers):
    # Save Q projection LoRA weights
    if hasattr(layer.self_attn.q_proj, 'A'):
        lora_state_dict[f'layer_{i}.self_attn.q_proj.A'] = layer.self_attn.q_proj.A.detach().cpu()
        lora_state_dict[f'layer_{i}.self_attn.q_proj.B'] = layer.self_attn.q_proj.B.detach().cpu()
    
    # Save V projection LoRA weights
    if hasattr(layer.self_attn.v_proj, 'A'):
        lora_state_dict[f'layer_{i}.self_attn.v_proj.A'] = layer.self_attn.v_proj.A.detach().cpu()
        lora_state_dict[f'layer_{i}.self_attn.v_proj.B'] = layer.self_attn.v_proj.B.detach().cpu()

# Save metadata
lora_state_dict['config'] = {
    'lora_rank': lora_rank,
    'lora_alpha': lora_alpha,
    'max_ctx_length': max_ctx_length
}

# Save to file
torch.save(lora_state_dict, f"../models/{run_name}_lora_weights.pt")
torch.save(optimizer.state_dict(), f"../models/{run_name}_optimizer_state.pth")
torch.save(scheduler.state_dict(), f"../models/{run_name}_scheduler_state.pth")


train_losses = np.array(train_losses)
val_losses = np.array(val_losses)

results = {}

results['train_losses'] = train_losses
results['val_losses'] = val_losses

torch.cuda.empty_cache()
accelerator = Accelerator()
model, test_loader = accelerator.prepare(model, test_loader)

model.eval()
with torch.no_grad():    
    for (batch,) in tqdm(test_loader):
        outputs = model.generate(batch, attention_mask = torch.ones_like(batch), max_new_tokens=max_ctx_length*2)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_decoded = decoding(prediction)
        break

test_decoded = decoding(test_texts_all[0])

results['prediction_decoded'] = prediction_decoded
results['test_decoded'] = test_decoded

running_mse_prey = running_mse(prediction_decoded[0], test_decoded[0])
running_mse_predator = running_mse(prediction_decoded[1], test_decoded[1])

pred_prey = prediction_decoded[0]
pred_predator = prediction_decoded[1]
actual_prey = test_decoded[0]
actual_predator = test_decoded[1]

mean_prey_squared_error = mean_squared_error(pred_prey, actual_prey)
mean_prey_absolute_error = mean_absolute_error(pred_prey, actual_prey)
r2_prey_score = r2_score(pred_prey, actual_prey)
mean_predator_squared_error = mean_squared_error(pred_predator, actual_predator)
mean_predator_absolute_error = mean_absolute_error(pred_predator, actual_predator)
r2_predator_score = r2_score(pred_predator, actual_predator)

results['metrics'] = {
    "mean_prey_squared_error": mean_prey_squared_error,
    "mean_prey_absolute_error": mean_prey_absolute_error,
    "r2_prey_score": r2_prey_score,
    "mean_predator_squared_error": mean_predator_squared_error,
    "mean_predator_absolute_error": mean_predator_absolute_error,
    "r2_predator_score": r2_predator_score,
    "running_mse_prey": running_mse_prey,
    "running_mse_predator": running_mse_predator,
}

results_cpu = move_to_cpu(results)
torch.save(results_cpu, f"../results/lora_run_{run_name}.pt")

# joblib.dump(results, f"../results/lora_run_{wandb.run.name}.gz", compress=3)