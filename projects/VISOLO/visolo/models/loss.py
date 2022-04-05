import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit

class CategoryLoss(nn.Module):
    """
    The Focal loss for category loss
    """
    def __init__(self, alpha=0.25, gamma=2., reduction='none'):
        super(CategoryLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        # pred, target = B, C, S, S or B, C, N, S, S
        loss = 0.

        if len(target) == 0:
            return None

        if len(target.size()) > 4:
            B,_,N,_,_ = target.size()
            num_foreground = target.sum(dim=(1,3,4))    # B,N
            num_foreground.clamp_min_(1.)
            loss = sigmoid_focal_loss_jit(pred, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
            loss = loss.sum(dim=(1,3,4))
            loss /= num_foreground
            loss = loss.mean()

        else:
            B,_,_,_ = target.size()
            num_foregrounds = target.reshape(B, -1).sum(dim=1)
            num_foregrounds.clamp_min_(1.)
            loss = sigmoid_focal_loss_jit(pred, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
            loss = loss.reshape(B, -1).sum(dim=1)
            loss /= num_foregrounds
            loss = loss.mean()

        return loss


class MaskLoss(nn.Module):
    """
    The Dice loss for mask loss
    Input pred_mask, pred_kernel and GT masks
    """
    def __init__(self, eps=1e-7, weight=3.):
        super(MaskLoss, self).__init__()
        self.eps = eps
        self.weight = weight

    def getMaskMap(self, mask, kernel, idxs):
        # mask: dim, H/4, W/4
        # kernel: dim, S1, S2

        C, S1, S2 = kernel.size()
        _, H, W = mask.size()
        device = mask.device

        mask_map = torch.zeros((len(idxs), H, W), dtype=torch.float32, device=device)
        mask = mask.unsqueeze(0)    # 1, dim, H/4, W/4
        for i in range(len(idxs)):
            idx = idxs[i]
            row = torch.div(idx, S2, rounding_mode='floor')
            col = idx % S2
            mask_map[i] = F.conv2d(mask, kernel[:,row,col].view(1,C,1,1)).squeeze()

        mask_map = mask_map.unsqueeze(0)
        mask_map = F.interpolate(mask_map, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)
        mask_map = torch.sigmoid(mask_map)

        return mask_map

    def forward(self, pred_mask, pred_kernel, target):
        # pred_mask: B, dim, N, H/4, W/4
        # pred_kernel: B, dim, N, S1, S2
        # target: B, S1*S2, N, H, W
        loss = 0.
        B, _, N, S1, S2 = pred_kernel.size()
        valid_location = target.sum(dim=(3,4))  # B, S**2, N
        valid_location[valid_location>0] = 1.

        for n in range(N):
            n_loss = 0.
            for b in range(B):

                idxs = torch.nonzero(valid_location[b,:,n], as_tuple=True)[0]
                if len(idxs) == 0:
                    n_loss += pred_mask[[]].sum() * pred_kernel[[]].sum() * 0.0
                    continue
                pred_b = self.getMaskMap(pred_mask[b,:,n,:,:], pred_kernel[b,:,n,:,:], idxs)   # len(idxs), H, W

                target_b = target[b,idxs,n,:,:]
                numerator = (pred_b * target_b).sum(dim=(1,2)) * 2   # len(idxs)
                denominator = (pred_b**2).sum(dim=(1,2)) + (target_b**2).sum(dim=(1,2)) # len(idxs)
                b_loss = torch.sum((1-numerator / (denominator + self.eps))) / max(1, len(idxs))
                n_loss += b_loss
            n_loss /= max(1, (valid_location[:,:,n].sum(dim=1) > 0).sum())
            loss += n_loss

        loss /= max(1, (valid_location.sum(dim=(0,1)) > 0).sum())

        loss *= self.weight

        return loss


class TrackingLoss(nn.Module):
    """
    The focal loss for Tracking loss
    """
    def __init__(self, alpha=0.25, gamma=2., reduction='none'):
        super(TrackingLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target, gt_cat=None):
        # pred, target: B, 1(N-1), S**2, S**2
        # gt_cat: B, C, S, S

        if gt_cat is not None:
            B,_,S,_ = gt_cat.size()
            _,N,S2,_ = target.size()
            loss = 0.

            valid_location = gt_cat.sum(dim=1).reshape(B, -1)  # B,C,S,S -> B,S,S -> B,S**2(row_major)
            valid_location[valid_location != 0] = 1.
            valid_location = valid_location.unsqueeze(1).expand(B,N,S2)  # B,S**2 -> B,1,S**2 -> B,N-1,S**2
            loss = sigmoid_focal_loss_jit(pred, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)   # B,N-1,S**2,S**2
            loss = torch.sum(loss.sum(dim=3) * valid_location, 2)  # B,N-1,S**2 -> B,N-1
            valid_location = valid_location.sum(dim=2)
            loss /= valid_location.clamp_min(1.) # B, N-1
            loss = loss.sum() / max(1, (valid_location>0).sum())

            return loss

        else:
            loss = 0.
            B,_,S2,_ = target.size()
            valid_location = target.sum(dim=3)  # B, 1, S**2
            valid_location[valid_location != 0] = 1.
            loss = sigmoid_focal_loss_jit(pred, target, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)   # B,N-1,S**2,S**2
            loss = torch.sum(loss.sum(dim=3) * valid_location, 2)   # B, 1
            valid_location = valid_location.sum(dim=2)  # B, 1
            loss /= valid_location.clamp_min(1.)
            loss = loss.sum() / max(1, (valid_location>0).sum())

            return loss



