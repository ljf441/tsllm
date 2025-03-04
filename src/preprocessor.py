import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib
import argparse

def get_dataset(system_id=0, points=1000):

    with h5py.File("lotka_volterra_data.h5", "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
        prey = trajectories[system_id, :points, 0]
        predator = trajectories[system_id, :points, 1]
        times = time_points[:points]

    return prey, predator, times

def load_qwen():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    # Freeze all parameters except LM head bias
    for param in model.parameters():
        param.requires_grad = False

    # Add trainable bias to logits
    assert model.lm_head.bias is None
    model.lm_head.bias = torch.nn.Parameter(
        torch.zeros(model.config.vocab_size, device=model.device)
    )
    model.lm_head.bias.requires_grad = True

    return model, tokenizer

def alpha_scaler(data, alpha, decimals=3):
    data = np.array(data)
    alpha_percentile = np.percentile(data, alpha)
    rescale = data/alpha_percentile
    return np.round(rescale, decimals = decimals)   

def encoding(prey, predator):
    series = np.column_stack((prey, predator))
    encoded = ';'.join([','.join(map(str, row)) for row in series])
    return encoded

def decoding(data):
    time_steps = data.split(';')
    decoded = np.array([list(map(float, step.split(','))) for step in time_steps])
    prey = decoded[:, 0]
    predator = decoded[:, 1]
    return prey, predator

model, tokenizer = load_qwen()

def process_data(system_id=0, points=1000, alpha=40, decimals=3):
    prey, predator, times = get_dataset(system_id=system_id, points=points)
    new_prey = alpha_scaler(prey, alpha=alpha, decimals=decimals)
    new_predator = alpha_scaler(predator, alpha=alpha, decimals=decimals)
    encoded = encoding(new_prey, new_predator)
    tokenized_data = tokenizer(encoded, return_tensors="pt")
    return tokenized_data, encoded, np.column_stack((prey, predator, new_prey, new_predator)), times

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocess lotka-volterra data")
    parser.add_argument("-s", "--system_id", type=int, default=0, help="system id")
    parser.add_argument("-p", "--points", type=int, default=1000, help="number of data points")
    parser.add_argument("-a", "--alpha", type=int, default=40, help="alpha scaling factor")
    parser.add_argument("-d", "--decimals", type=int, default=3, help="number of decimal places to round")

    args = parser.parse_args()

    print("EXAMPLE (SYSTEM_ID=0, POINTS=3):")
    tokenized_data, preprocessed_data, combined_data, times = process_data(points=3)
    print("Preprocessed data:", preprocessed_data)
    print("Tokenized results:", tokenized_data["input_ids"].tolist()[0])

    tokenized_data, preprocessed_data, combined_data, times = process_data(system_id=args.system_id, points=args.points, alpha=args.alpha, decimals=args.decimals)
    joblib.dump(tokenized_data, "tokenized_data.pkl")