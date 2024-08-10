import torch


# @torch.jit.script
def non_overlap_mse(pred, target, radius):
    pred = pred.squeeze(1)
    height, width = pred.shape[1], pred.shape[2]
    if radius is None:
        radius = torch.tensor(height // 2)
    else:
        radius = torch.tensor(radius)
    # get the cropped region
    pred_core = pred[:, height // 2 - radius: height // 2 + radius + 1,
                     width // 2 - radius: width // 2 + radius + 1]
    target_core = target[:, height // 2 - radius: height // 2 + radius + 1,
                         width // 2 - radius: width // 2 + radius + 1]

    core_valid_mask = (target_core >= 0).detach()
    nums = torch.sum(core_valid_mask)
    v = torch.sum((pred_core[core_valid_mask] - target_core[core_valid_mask]) ** 2)

    return v, nums


