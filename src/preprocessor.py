import h5py
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import joblib
import argparse
from sklearn.model_selection import train_test_split

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

def scaler(prey, predator, alpha, decimals=3):
    prey = np.array(prey)
    predator = np.array(predator)
    data = np.stack([prey, predator], axis=-1)
    rescaled = data/alpha * 10
    return np.round(rescaled, decimals = decimals)

def encoding(prey, predator):
    series = np.column_stack((prey, predator))
    encoded = ';'.join([','.join(map(str, row)) for row in series])
    return encoded

def decoding(data):
    time_steps = data.split(';')
    decoded = np.array([list(map(float, step.split(','))) 
                        for step in time_steps 
                        if step.strip() 
                        and len(step.split(',')) == 2 
                        and all(value.strip() for value in step.split(','))]
                        [:100])
    prey = decoded[:, 0]
    predator = decoded[:, 1]
    return prey, predator

model, tokenizer = load_qwen()

def get_and_process_data(system_id=0, points=100, alpha=5, decimals=3):
    prey, predator, times = get_dataset(system_id=system_id, points=points)
    scaled = scaler(prey, predator, alpha = alpha, decimals=decimals)
    new_prey, new_predator = scaled[:, 0], scaled[:, 1] 
    encoded = encoding(new_prey, new_predator)
    tokenized_data = tokenizer(encoded, return_tensors="pt")
    return tokenized_data, encoded, np.column_stack((prey, predator, new_prey, new_predator)), times

def load_and_preprocess(file_path, test_size=0.2, alpha=5, decimals=3):
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]
        prey = trajectories[:, :, 0]
        predator = trajectories[:, :, 1]

    scaled = scaler(prey, predator, alpha=alpha, decimals=decimals)
    new_prey = scaled[:, :, 0]
    new_predator = scaled[:, :, 1]

    stacked_data = np.stack((new_prey, new_predator), axis=-1)

    train_data, temp_data = train_test_split(stacked_data, test_size=test_size, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True)

    data = []

    for d in [train_data, val_data, test_data]:
        prey, predator = d[:, :, 0], d[:, :, 1]
        encoded = [encoding(prey, predator) for prey, predator in zip(prey, predator)]
        data.append(encoded)

    return data

def process_data(texts, tokenizer, points=80):
    given_input_ids = []
    for text in texts:
        given_text = ';'.join([chunk for i, chunk in enumerate(text.split(';')) if i < points])
        encoding_given = tokenizer(given_text, return_tensors="pt", padding='max_length', padding_side='left', max_length=1200)
        given_input_ids.append(encoding_given.input_ids[0])
    return np.stack([text for text in texts]), torch.stack(given_input_ids)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocess lotka-volterra data")
    parser.add_argument("-s", "--system_id", type=int, default=0, help="system id")
    parser.add_argument("-p", "--points", type=int, default=1000, help="number of data points")
    parser.add_argument("-a", "--alpha", type=int, default=40, help="scaling factor")
    parser.add_argument("-d", "--decimals", type=int, default=3, help="number of decimal places to round")

    args = parser.parse_args()

    print("EXAMPLE (SYSTEM_ID=0, POINTS=3):")
    tokenized_data, preprocessed_data, combined_data, times = process_data(points=3)
    print("Preprocessed data:", preprocessed_data)
    print("Tokenized results:", tokenized_data["input_ids"].tolist()[0])

    tokenized_data, preprocessed_data, combined_data, times = process_data(system_id=args.system_id, points=args.points, alpha=args.alpha, decimals=args.decimals)
    joblib.dump(tokenized_data, "tokenized_data.pkl")