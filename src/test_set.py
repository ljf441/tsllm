import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from accelerate import Accelerator

from collections.abc import Iterable

from preprocessor import load_and_preprocess, decoding, get_dataset
from qwen import load_qwen

import numpy as np

import matplotlib.pyplot as plt

import joblib

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

import pandas as pd

def running_mse(prediction, actual):
    """
    Calculate running MSE comparing the prediction and actual values.

    Args:
        prediction (ndarray): Predicted values.
        actual (ndarray): Actual values.

    Returns:
        mse (list): List of running MSE values.
    """
    mse = []
    for i in range(len(prediction)):
        mse.append(mean_squared_error(prediction[:i+1], actual[:i+1]))
    return mse

def get_metrics(actual_preys, actual_predators, pred_preys, pred_predators):
    """
    Calculate metrics for predator and prey time series data comparing predicted and actual data.

    Args:
        actual_preys (ndarray): Actual prey values.
        actual_predators (ndarray): Actual predator values.
        pred_preys (ndarray): Predicted prey values.
        pred_predators (ndarray): Predicted predator values.
    
    Returns:
        metrics (list): List of tuples containing calculated metrics for prey and predator.
    """
    metrics = []
    for actual_prey, actual_predator, pred_prey, pred_predator in zip(actual_preys, actual_predators, pred_preys, pred_predators):
        mse_prey = mean_squared_error(actual_prey[80:], pred_prey[80:])
        mse_predator = mean_squared_error(actual_predator[80:], pred_predator[80:])
        mae_prey = mean_absolute_error(actual_prey[80:], pred_prey[80:])
        mae_predator = mean_absolute_error(actual_predator[80:], pred_predator[80:])
        mape_prey = mean_absolute_percentage_error(actual_prey[80:], pred_prey[80:])
        mape_predator = mean_absolute_percentage_error(actual_predator[80:], pred_predator[80:])
        rmse_prey = root_mean_squared_error(actual_prey[80:], pred_prey[80:])
        rmse_predator = root_mean_squared_error(actual_predator[80:], pred_predator[80:])
        running_mse_prey = running_mse(pred_prey[80:], actual_prey[80:])
        running_mse_predator = running_mse(pred_predator[80:], actual_predator[80:])
        metrics.append((mse_prey, mse_predator, mae_prey, mae_predator, mape_prey, mape_predator, rmse_prey, rmse_predator, running_mse_prey, running_mse_predator))
    return metrics

def move_to_cpu(obj):
    """
    Recursively convert tensors to CPU NumPy arrays.
    
    Args:
        obj (any): Input object (tensor, NumPy array, list, dict, etc.).
    Returns:
        obj (any): Object moved to CPU.
    """
    if isinstance(obj, torch.Tensor):
        # Move tensor to CPU and convert to NumPy
        return obj.detach().cpu().numpy()
    
    elif isinstance(obj, np.ndarray):
        # Already a NumPy array (no action needed)
        return obj
    
    elif isinstance(obj, dict):
        # Process dictionary values and recursively move to cpu
        return {k: move_to_cpu(v) for k, v in obj.items()}
    
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        # Process lists, tuples, etc.
        return type(obj)(move_to_cpu(v) for v in obj)
    
    else:
        # Return Python primitives (int, float, etc.) as-is
        return obj
    
def print_metrics(metrics):
    """
    Prints the metrics in a formatted table.

    Args:
        metrics (list): List of tuples containing calculated metrics for prey and predator.
        system_ids (list): List of system IDs corresponding to the metrics.

    Returns:
        None
    """
    df = pd.DataFrame(metrics, columns=["MSE Prey", "MSE Predator", "MAE Prey", "MAE Predator", "MAPE Prey", "MAPE Predator", "RMSE Prey", "RMSE Predator", "Running MSE Prey", "Running MSE Predator"])
    df.set_index("System ID", inplace=True)
    df = df.round(4)
    df.drop(columns=["Running MSE Prey", "Running MSE Predator"], inplace=True)
    return df
    
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

test_size = 0.2
points = 80

# Process the data into sequences of text
np.random.seed(442)
_, _, test_texts = load_and_preprocess("lotka_volterra_data.h5", test_size=test_size)

def process_data(texts, tokenizer, points=80):
    """
    Process test data into sequences of text and tokenize them.

    Args:
        texts (list): List of text sequences.
        tokenizer (Tokenizer): Tokenizer to use for processing.
        points (int): Number of points to consider.

    Returns:
        tuple: Tuple containing the processed texts and their corresponding tokenized input IDs.
    """
    given_input_ids = []
    for text in texts:
        given_text = ';'.join([chunk for i, chunk in enumerate(text.split(';')) if i < points])
        encoding_given = tokenizer(given_text, return_tensors="pt", padding='max_length', padding_side='left', max_length=1200)
        given_input_ids.append(encoding_given.input_ids[0])
    return np.stack([text for text in texts]), torch.stack(given_input_ids)

full_times = get_dataset("lotka_volterra_data.h5")[-1]

# UNTRAINED MODEL

# initialise accelerator
accelerator = Accelerator()

model, tokenizer = load_qwen()
tokenizer.padding_side = 'left'

# tokenize the test set
test_texts_all, test_input_ids_some = process_data(
    test_texts, tokenizer, points=points
)

# put test set into DataLoader
test_dataset = TensorDataset(test_input_ids_some)
test_loader = DataLoader(test_dataset, shuffle=False)

# prepare accelerator with test set
model, test_loader = accelerator.prepare(model, test_loader)

# predict specific systems
prediction_decodeds = []
test_decodeds = []
model.eval()
with torch.no_grad():    
    for idx, (batch,) in enumerate(tqdm(test_loader)):
        outputs = model.generate(batch, attention_mask = torch.ones_like(batch), max_new_tokens=1300)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_decoded = decoding(prediction)
        prediction_decodeds.append(prediction_decoded)
        test_text = test_texts_all[idx]
        test_decoded = decoding(test_text)
        test_decodeds.append(test_decoded)

test_decodeds = np.array(test_decodeds)
prediction_decodeds = np.array(prediction_decodeds)

actual_preys = move_to_cpu(test_decodeds[:, 0])
actual_predators = move_to_cpu(test_decodeds[:, 1])
pred_preys = move_to_cpu(prediction_decodeds[:, 0])
pred_predators = move_to_cpu(prediction_decodeds[:, 1])

torch.save((actual_preys, actual_predators, pred_preys, pred_predators, full_times), "full_untrained_predictions.pt")

# joblib.dump((actual_preys, actual_predators, pred_preys, pred_predators, full_times), "full_untrained_predictions.pkl")
# actual_preys, actual_predators, pred_preys, pred_predators, full_times = joblib.load("full_untrained_predictions.pkl")

metrics = get_metrics(actual_preys, actual_predators, pred_preys, pred_predators)

df = print_metrics(metrics)

joblib.dump(df, "full_untrained_metrics.pkl")

# DEFAULT MODEL

lora_rank = 4
lora_alpha = lora_rank
batch_size = 4
max_ctx_length = 512
    
# initialise accelerator
accelerator = Accelerator()

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

# Load the saved LoRA weights
model_name = "CSD3_3k_default_best"
saved_model_path = f"../models/{model_name}_lora_weights.pt"
lora_state_dict = torch.load(saved_model_path)
print(f"Loading LoRA weights from {saved_model_path}")

# Extract configuration
loaded_config = lora_state_dict.get('config', {})
print(f"Loaded model config: {loaded_config}")

# Load weights into model
for i, layer in enumerate(model.model.layers):
    # Load Q projection LoRA weights
    if f'layer_{i}.self_attn.q_proj.A' in lora_state_dict:
        layer.self_attn.q_proj.A.data = lora_state_dict[f'layer_{i}.self_attn.q_proj.A'].to(layer.self_attn.q_proj.A.device)
        layer.self_attn.q_proj.B.data = lora_state_dict[f'layer_{i}.self_attn.q_proj.B'].to(layer.self_attn.q_proj.B.device)
    
    # Load V projection LoRA weights
    if f'layer_{i}.self_attn.v_proj.A' in lora_state_dict:
        layer.self_attn.v_proj.A.data = lora_state_dict[f'layer_{i}.self_attn.v_proj.A'].to(layer.self_attn.v_proj.A.device)
        layer.self_attn.v_proj.B.data = lora_state_dict[f'layer_{i}.self_attn.v_proj.B'].to(layer.self_attn.v_proj.B.device)

print("LoRA weights loaded successfully")

# tokenize the test set
test_texts_all, test_input_ids_some = process_data(
    test_texts, tokenizer, points=points
)

# put test set into DataLoader
test_dataset = TensorDataset(test_input_ids_some)
test_loader = DataLoader(test_dataset, shuffle=False)

# prepare accelerator with test set
model, test_loader = accelerator.prepare(model, test_loader)

# predict specific systems
prediction_decodeds = []
test_decodeds = []
model.eval()
with torch.no_grad():    
    for idx, (batch,) in enumerate(tqdm(test_loader)):
        outputs = model.generate(batch, attention_mask = torch.ones_like(batch), max_new_tokens=1300)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_decoded = decoding(prediction)
        prediction_decodeds.append(prediction_decoded)
        test_text = test_texts_all[idx]
        test_decoded = decoding(test_text)
        test_decodeds.append(test_decoded)

test_decodeds = np.array(test_decodeds)
prediction_decodeds = np.array(prediction_decodeds)

actual_preys = move_to_cpu(test_decodeds[:, 0])
actual_predators = move_to_cpu(test_decodeds[:, 1])
pred_preys = move_to_cpu(prediction_decodeds[:, 0])
pred_predators = move_to_cpu(prediction_decodeds[:, 1])

torch.save((actual_preys, actual_predators, pred_preys, pred_predators, full_times), "full_default_predictions.pt")

metrics = get_metrics(actual_preys, actual_predators, pred_preys, pred_predators)

df = print_metrics(metrics)

joblib.dump(df, "full_default_metrics.pkl")

# FINAL MODEL

# load hyperparameters from file
file_path = "../results/best_overall_params.pt"
best_overall_params = torch.load(file_path, weights_only=False, map_location=torch.device('cpu'))

lora_rank = best_overall_params['lora_rank']
lora_alpha = lora_rank
batch_size = 4
learning_rate = best_overall_params['learning_rate']
max_ctx_length = best_overall_params['max_ctx_length']
    
# initialise accelerator
accelerator = Accelerator()

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

# Load the saved LoRA weights
model_name = "CSD3_15k_seed_defaults_best"
saved_model_path = f"../models/{model_name}_lora_weights.pt"
lora_state_dict = torch.load(saved_model_path)
print(f"Loading LoRA weights from {saved_model_path}")

# Extract configuration
loaded_config = lora_state_dict.get('config', {})
print(f"Loaded model config: {loaded_config}")

# Load weights into model
for i, layer in enumerate(model.model.layers):
    # Load Q projection LoRA weights
    if f'layer_{i}.self_attn.q_proj.A' in lora_state_dict:
        layer.self_attn.q_proj.A.data = lora_state_dict[f'layer_{i}.self_attn.q_proj.A'].to(layer.self_attn.q_proj.A.device)
        layer.self_attn.q_proj.B.data = lora_state_dict[f'layer_{i}.self_attn.q_proj.B'].to(layer.self_attn.q_proj.B.device)
    
    # Load V projection LoRA weights
    if f'layer_{i}.self_attn.v_proj.A' in lora_state_dict:
        layer.self_attn.v_proj.A.data = lora_state_dict[f'layer_{i}.self_attn.v_proj.A'].to(layer.self_attn.v_proj.A.device)
        layer.self_attn.v_proj.B.data = lora_state_dict[f'layer_{i}.self_attn.v_proj.B'].to(layer.self_attn.v_proj.B.device)

print("LoRA weights loaded successfully")

# tokenize the test set
test_texts_all, test_input_ids_some = process_data(
    test_texts, tokenizer, points=points
)

# put test set into DataLoader
test_dataset = TensorDataset(test_input_ids_some)
test_loader = DataLoader(test_dataset, shuffle=False)

# prepare accelerator with test set
model, test_loader = accelerator.prepare(model, test_loader)

# predict specific systems
prediction_decodeds = []
test_decodeds = []
model.eval()
with torch.no_grad():    
    for idx, (batch,) in enumerate(tqdm(test_loader)):
        outputs = model.generate(batch, attention_mask = torch.ones_like(batch), max_new_tokens=1300)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        prediction_decoded = decoding(prediction)
        prediction_decodeds.append(prediction_decoded)
        test_text = test_texts_all[idx]
        test_decoded = decoding(test_text)
        test_decodeds.append(test_decoded)

test_decodeds = np.array(test_decodeds)
prediction_decodeds = np.array(prediction_decodeds)

actual_preys = move_to_cpu(test_decodeds[:, 0])
actual_predators = move_to_cpu(test_decodeds[:, 1])
pred_preys = move_to_cpu(prediction_decodeds[:, 0])
pred_predators = move_to_cpu(prediction_decodeds[:, 1])

torch.save((actual_preys, actual_predators, pred_preys, pred_predators, full_times), "full_final_predictions.pt")

metrics = get_metrics(actual_preys, actual_predators, pred_preys, pred_predators)

df = print_metrics(metrics)

joblib.dump(df, "full_final_metrics.pkl")