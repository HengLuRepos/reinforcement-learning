import torch
import torch.nn as nn
import numpy as np
import logging

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def build_mlp(observation_dim,action_dim,layer_size,n_layers):
    return nn.Sequential(
        nn.Linear(observation_dim,layer_size),
        nn.ReLU(),
        *([nn.Linear(layer_size,layer_size), nn.ReLU()]*(n_layers - 1)),
        nn.Linear(layer_size,action_dim)
    )

def np2torch(arr):
    x = torch.from_numpy(arr) if isinstance(arr, np.ndarray) else arr
    return x.to(device).float()

    
def get_logger(filename):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename,mode='w')
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger