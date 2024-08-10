import torch.nn as nn


class MaskedMAE(nn.Module):
    def __init__(self):
        super(MaskedMAE, self).__init__()
        self.l1_func = nn.L1Loss()

    def forward(self, preds, gts):
        if preds.shape != gts.shape:
            preds = preds.squeeze(1)
        valid_mask = (gts >= 0).detach()
        loss = self.l1_func(preds[valid_mask], gts[valid_mask])
        return loss


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.mse_func = nn.MSELoss()

    def forward(self, preds, gts):
        if preds.shape != gts.shape:
            preds = preds.squeeze(1)
        valid_mask = (gts >= 0).detach()
        loss = self.mse_func(preds[valid_mask], gts[valid_mask])
        return loss
