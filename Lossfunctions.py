import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    '''
    Implementation of the dice loss proposed in:
    https://arxiv.org/abs/1707.03237
    '''

    def __init__(self, smooth: float = 1.0) -> None:
        '''
        Constructor method
        :param smooth: (float) Smoothness factor used in computing the dice loss
        '''
        # Call super constructor
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        Forward method calculates the dice loss
        :param prediction: (torch.tensor) Prediction tensor including probabilities
        :param label: (torch.tensor) Label tensor (one-hot encoded)
        :return: (torch.tensor) Dice loss
        '''
        # Flatten prediction and label
        prediction = prediction.view(-1)
        label = label.view(-1)
        # Calc intersection
        intersect = torch.sum((prediction * label)) + self.smooth
        # Calc union
        union = torch.sum(prediction) + torch.sum(label) + self.smooth
        # Calc dice loss
        dice_loss = 1.0 - ((2.0 * intersect) / (union))
        return dice_loss


class FocalLoss(nn.Module):
    '''
    Implementation of the binary focal loss proposed in:
    https://arxiv.org/abs/1708.02002
    '''

    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduce: str = 'mean') -> None:
        '''
        Constructor method
        :param alpha: (float) Alpha constant (see paper)
        :param gamma: (float) Gamma constant (ses paper)
        :param reduce: (str) Reduction operation (mean, sum or none)
        '''
        # Call super constructor
        super(FocalLoss, self).__init__()
        # Check reduce parameter
        assert reduce in ['mean', 'sum', 'none'], 'Illegal value of reduce parameter. Use mean, sum or none.'
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce

    def forward(self, prediction: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        '''
        Forward method calculates the dice loss
        :param prediction: (torch.tensor) Prediction tensor including probabilities
        :param label: (torch.tensor) Label tensor (one-hot encoded)
        :return: (torch.tensor) Dice loss
        '''
        # Calc binary cross entropy loss
        cross_entropy_loss = F.binary_cross_entropy(prediction, label, reduction='none')
        # Calc focal loss
        focal_loss = self.alpha * (1.0 - prediction) ** self.gamma * cross_entropy_loss
        # Reduce loss
        if self.reduce == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduce == 'sum':
            focal_loss = torch.sum(focal_loss)
        return focal_loss


if __name__ == '__main__':
    dice_loss = DiceLoss()
    # input = torch.cat([torch.ones(1, 1, 256, 256), torch.zeros(1, 1, 256, 256)], dim=1) # torch.softmax(torch.randn([1, 2, 256, 256]), dim=1)
    # label = torch.cat([torch.ones(1, 1, 256, 256), torch.zeros(1, 1, 256, 256)], dim=1)
    input = torch.ones([1, 1000])
    label = torch.zeros([1, 1000])
    loss = dice_loss(input, label)
    print(loss)