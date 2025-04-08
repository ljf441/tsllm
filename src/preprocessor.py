import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split

def get_dataset(file_path, system_id=0, points=100):
    """
    Load the Lotka-Volterra dataset from an HDF5 file and extract the prey and predator data.

    Args:
        file_path (str): Path to the HDF5 file.
        system_id (int): The ID of the system to load.
        points (int): The number of points to load.
    Returns:
        prey (np.ndarray): The prey values.
        predator (np.ndarray): The predator values.
        times (np.ndarray): The time points.
    """
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]
        time_points = f["time"][:]
        prey = trajectories[system_id, :points, 0]
        predator = trajectories[system_id, :points, 1]
        times = time_points[:points]

    return prey, predator, times

def scaler(prey, predator, alpha, decimals=3):
    """
    Scales and rounds the prey and predator data.

    Args:
        prey (np.ndarray): The prey values.
        predator (np.ndarray): The predator values.
        alpha (float): Scaling factor.
        decimals (int): Number of decimal places for rounding.
    Returns:
        rounded (np.ndarray): The scaled prey and predator values.
    """
    prey = np.array(prey)
    predator = np.array(predator)
    data = np.stack([prey, predator], axis=-1)
    rescaled = data/alpha * 10
    rounded = np.round(rescaled, decimals=decimals)
    return rounded
def encoding(prey, predator):
    """
    Encode the prey and predator data into a string format. 

    Args:
        prey (np.ndarray): The prey values.
        predator (np.ndarray): The predator values.
    Returns:
        encoded (str): The encoded data string.
    """
    series = np.column_stack((prey, predator))
    encoded = ';'.join([','.join(map(str, row)) for row in series])
    return encoded

def scale_and_encode(prey, predator, alpha, decimals):
    """
    Scale and encode the prey and predator data.

    Args:
        prey (np.ndarray): The prey values.
        predator (np.ndarray): The predator values.
        alpha (float): Scaling factor.
        decimals (int): Number of decimal places for scaling.
    Returns:
        encoded (str): The encoded data string.
    """
    prey = np.array(prey)
    predator = np.array(predator)
    data = np.stack([prey, predator], axis=-1)
    rescaled = data/alpha * 10
    rescaled = np.round(rescaled, decimals=decimals)
    series = np.column_stack((rescaled[:, 0], rescaled[:, 1]))
    encoded = ';'.join([','.join(map(str, row)) for row in series])
    return encoded

def decoding(data):
    """
    Decode the encoded data back to prey and predator values, handling numeric extraction from strings.
    
    Args:
        data (str): The encoded data string containing potential text labels.
        
    Returns:
        prey (np.ndarray): The decoded prey values.
        predator (np.ndarray): The decoded predator values.
    """
    def extract_number(s):
        """Extract first numeric value from string using regex."""
        match = re.search(r"[-+]?\d*\.?\d+", s.strip())
        if match:
            return float(match.group())
        return float(-99)

    time_steps = data.split(';')
    
    decoded = np.array([
        list(map(extract_number, step.split(','))) 
        for step in time_steps 
        if step.strip()
        and len(step.split(',')) == 2
        and all(value.strip() for value in step.split(','))
    ][:100])
    
    prey = decoded[:, 0]
    predator = decoded[:, 1]
    return prey, predator

def get_and_process_data(file_path, tokenizer, system_id=0, points=100, alpha=5, decimals=3):
    """
    Load and preprocess the dataset, including scaling and encoding.
    
    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        system_id (int): The ID of the system to load.
        points (int): The number of points to load.
        alpha (float): Scaling factor.
        decimals (int): Number of decimal places for scaling.
        
    Returns:
        tokenized_data (dict): Tokenized data.
        encoded (str): Encoded data.
        combined_data (np.ndarray): Data of original and scaled numerical values.
        times (np.ndarray): Time points.
    """
    prey, predator, times = get_dataset(file_path, system_id=system_id, points=points)
    scaled = scaler(prey, predator, alpha = alpha, decimals=decimals)
    new_prey, new_predator = scaled[:, 0], scaled[:, 1] 
    encoded = encoding(new_prey, new_predator)
    tokenized_data = tokenizer(encoded, return_tensors="pt")
    return tokenized_data, encoded, np.column_stack((prey, predator, new_prey, new_predator)), times

def load_and_preprocess(file_path, test_size=0.2, alpha=5, decimals=3, seed=442):
    """
    Load and preprocess the dataset from an HDF5 file, applying scaling and encoding.
    Args:
        file_path (str): Path to the HDF5 file.
        test_size (float): Proportion of the dataset to include in the test split.
        alpha (float): Scaling factor.
        decimals (int): Number of decimal places for scaling.
    Returns:
        data (list): List containing train, validation, and test datasets.
    """
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]
        prey = trajectories[:, :, 0]
        predator = trajectories[:, :, 1]

    scaled = scaler(prey, predator, alpha=alpha, decimals=decimals)
    new_prey = scaled[:, :, 0]
    new_predator = scaled[:, :, 1]

    stacked_data = np.stack((new_prey, new_predator), axis=-1)

    np.random.seed(seed)

    train_data, temp_data = train_test_split(stacked_data, test_size=test_size, shuffle=True)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True)

    data = []

    for d in [train_data, val_data, test_data]:
        prey, predator = d[:, :, 0], d[:, :, 1]
        encoded = [encoding(prey, predator) for prey, predator in zip(prey, predator)]
        data.append(encoded)

    return data

def load_and_process_example(file_path, tokenizer, points=100, test_size=0.2, alpha=5, decimals=3, seed=442, id=0):
    """
    Load and preprocess a specific example from the dataset, applying scaling and encoding.

    Args:
        file_path (str): Path to the HDF5 file.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        points (int): The number of points to load.
        test_size (float): Proportion of the dataset to include in the test split.
        alpha (float): Scaling factor.
        decimals (int): Number of decimal places for scaling.
    Returns:
        tokenized_data (torch.Tensor): Tokenized data.
        encoded (str): Encoded data.
        combined_data (np.ndarray): Data of original and scaled numerical values.
        time (np.ndarray): Time points.
    """
    with h5py.File(file_path, "r") as f:
        trajectories = f["trajectories"][:]
        prey = trajectories[:, :, 0]
        predator = trajectories[:, :, 1]
        time = f["time"][:]

    scaled = scaler(prey, predator, alpha=alpha, decimals=decimals)
    new_prey = scaled[:, :, 0]
    new_predator = scaled[:, :, 1]

    stacked_data = np.stack((new_prey, new_predator), axis=-1)
    stacked = np.stack((prey, predator), axis=-1)

    np.random.seed(seed)

    _, temp_data = train_test_split(stacked_data, test_size=test_size, shuffle=True)
    _, test_data = train_test_split(temp_data, test_size=0.5, shuffle=True)

    np.random.seed(seed)

    _, temp = train_test_split(stacked, test_size=test_size, shuffle=True)
    _, test = train_test_split(temp, test_size=0.5, shuffle=True)

    prey, predator = test[id, :, 0], test[id, :, 1]
    new_prey, new_predator = test_data[id, :, 0], test_data[id, :, 1]

    encoded = encoding(test_data[id, :, 0], test_data[id, :, 1])
    given_text = ';'.join([chunk for i, chunk in enumerate(encoded.split(';')) if i < points])
    tokenized_data = tokenizer(given_text, return_tensors="pt")
    
    return tokenized_data, given_text, np.column_stack((prey, predator, new_prey, new_predator)), time

def process_data(texts, tokenizer, points=80):
    """
    Process the input texts by tokenizing and padding them.

    Args:
        texts (list): List of input texts.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer to use.
        points (int): The number of points to load.
    Returns:
        texts (np.ndarray): The processed texts.
        given_input_ids (torch.Tensor): The tokenized and padded input IDs.
    """
    given_input_ids = []
    for text in texts:
        given_text = ';'.join([chunk for i, chunk in enumerate(text.split(';')) if i < points])
        encoding_given = tokenizer(given_text, return_tensors="pt", padding='max_length', padding_side='left', max_length=1200)
        given_input_ids.append(encoding_given.input_ids[0])
    return np.stack([text for text in texts]), torch.stack(given_input_ids)