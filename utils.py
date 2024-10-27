import os 
import torch
from torch import nn
from safetensors.torch import load_model

def load_last_ckpt(model_dir_path: str,
                     model: torch.nn.Module,
                     device: str):
     """
     Load the last checkpoint of the model from given dir.
     Args:
          model_dir_path: str, the path of the model directory.
          model: torch.nn.Module, the model to load the checkpoint.
          device: str, the device to load the model.
     """
     ckpt_list = os.listdir(model_dir_path)
     ckpt_list = [int(ckpt.split('-')[1]) for ckpt in ckpt_list if ckpt.startswith('checkpoint')]
     if ckpt_list == []:
          raise FileNotFoundError(f"No checkpoint found in {model_dir_path}!")
     ckpt_list.sort()
     last_ckpt = ckpt_list[-1]
     ckpt_path = os.path.join(model_dir_path, f'checkpoint-{last_ckpt}/model.safetensors')
     load_model(model, ckpt_path, device=device)


def count_params(model: nn.Module):
    """
    Count the number of parameters in the model.
    Args:
        model: nn.Module, the model to count the parameters.
    Returns:
        n_params: int, the number of parameters in the model.
    """
    n_params = sum(p.numel() for p in model.parameters())
    return n_params