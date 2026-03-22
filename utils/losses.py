import torch
import torch.nn.functional as F

def reconstruction_loss(pred, target):
    loss = F.mse_loss(pred, target)
    return loss
def anomaly_score(pred, target):
    error = torch.mean(
        (pred-target)**2,
        dim=[1, 2, 3]
    )
    return error
