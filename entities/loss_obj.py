from torch import nn


class LossObj:
    def __init__(self, loss_name='mse'):
        if loss_name == 'mse':
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError("Loss function not supported")

    def get_loss(self):
        return self.loss_func
