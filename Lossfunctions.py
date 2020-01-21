import torch


def dice_loss(prediction: torch.tensor, label: torch.tensor, smooth: float = 1.0) -> torch.tensor:
    """
    Implementation of the differentiable dice loss
    :param prediction: (torch.tensor) Prediction tensor
    :param label: (torch.tensor) Label tensor
    :param smooth: (float) Smoothness factor of the dice loss
    :return:
    """
    # Flatten prediction and label
    prediction = prediction.view(-1)
    label = label.view(-1)
    # Calc loss
    intersect = torch.sum((prediction * label) + smooth)
    union = torch.sum(pred) + torch.sum(label) + smooth
    loss = 1.0 - ((2.0 * intersect) / (union))
    return loss
