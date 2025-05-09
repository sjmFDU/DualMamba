
import torch
from models.get_model import get_model

model = get_model(1, 'ssftt', 'ip', 9)
total_params = sum(p.numel() for p in model.parameters())
print(total_params)