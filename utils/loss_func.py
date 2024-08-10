import torch.nn as nn
# from utils.ssim import gt_guided_mask_ssim


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

'''
class MaskedMAE_SSMI(nn.Module):
    def __init__(self, weight_mae=0.5):
        super(MaskedMAE_SSMI, self).__init__()
        self.w_mae = weight_mae
        self.l1loss = MaskedMAE()

    def forward(self, x, y):
        ssim = gt_guided_mask_ssim(X=x.clone(), Y=y.clone(), data_range=25)
        l1_loss = self.l1loss(x, y)
        loss = (1 - ssim) * (1 - self.w_mae) + self.w_mae * l1_loss
        return loss


class MaskedMSE_SSMI(nn.Module):
    def __init__(self, weight_mse=0.5):
        super(MaskedMSE_SSMI, self).__init__()
        self.w_mse = weight_mse
        self.l2loss = MaskedMSE()

    def forward(self, x, y):
        ssim = gt_guided_mask_ssim(X=x.clone(), Y=y.clone(), data_range=25)
        l2_loss = self.l2loss(x, y)
        loss = (1 - ssim) * (1 - self.w_mse) + self.w_mse * l2_loss
        return loss
'''