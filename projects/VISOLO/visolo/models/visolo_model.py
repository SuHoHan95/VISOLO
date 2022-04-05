from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys

from detectron2.layers import Conv2d, get_norm

from .loss import *

def init_msra_fill(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def addCoord(r):
    B, _, H, W = r.size()

    h_coord = torch.linspace(-1, 1, steps=H)
    w_coord = torch.linspace(-1, 1, steps=W)
    coord_y, coord_x = torch.meshgrid(h_coord, w_coord)  # (H,W)
    coord_yx = torch.cat((coord_y[None, :], coord_x[None, :]), dim=0)  # (2,H,W)
    coord_yx = coord_yx.unsqueeze(0).repeat(B, 1, 1, 1)  # (B,2,H,W)
    cat_r_coord = torch.cat((r, coord_yx.cuda()), dim=1)  # (B,C+2,H,W)

    return cat_r_coord


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1, conv_norm=""):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = Conv2d(indim, outdim, kernel_size=3, stride=stride, padding=1,
                                     norm=get_norm(conv_norm, outdim))

        self.conv1 = Conv2d(indim, outdim, kernel_size=3, stride=stride, padding=1,
                            norm=get_norm(conv_norm, outdim), activation=nn.ReLU())
        self.conv2 = Conv2d(outdim, outdim, kernel_size=3, stride=stride, padding=1,
                            norm=get_norm(conv_norm, outdim))

        init_msra_fill(self)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(r)

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, indim, outdim, conv_norm="", scale_factor=2):
        super(Refine, self).__init__()
        self.convSCF = Conv2d(indim, outdim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, outdim))
        self.ResSCF = ResBlock(outdim, outdim, stride=1, conv_norm=conv_norm)
        self.ResF = ResBlock(outdim, outdim, stride=1, conv_norm=conv_norm)
        self.scale_factor = scale_factor
        init_msra_fill(self)

    def forward(self, scf, f):
        s = self.ResSCF(self.convSCF(scf))
        sum_f = s + F.interpolate(f, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        sum_f = self.ResF(sum_f)

        return sum_f


class KernelBranch(nn.Module):
    def __init__(self, S, indim=1024, outdim=256, conv_norm=""):
        super(KernelBranch, self).__init__()

        self.S = S
        self.conv_layers = nn.Sequential()
        self.conv_layers_post = nn.Sequential()

        cur_channel = indim + 2
        for k in range(3):
            conv = Conv2d(
                cur_channel,
                outdim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=get_norm(conv_norm, outdim),
                activation=nn.ReLU(),
            )
            self.conv_layers.add_module("kernel_conv{}".format(k+1), conv)
            cur_channel = outdim

        for k in range(2):
            conv = Conv2d(
                outdim,
                outdim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=get_norm(conv_norm, outdim),
                activation=nn.ReLU(),
            )
            self.conv_layers_post.add_module("kernel_conv{}".format(k+4), conv)

        self.pred = Conv2d(outdim, outdim, kernel_size=1, padding=0, stride=1, norm=get_norm(conv_norm, outdim))
        init_msra_fill(self)

    def forward(self, f, pre=True):
        if pre:
            p = self.conv_layers(F.interpolate(addCoord(f), size=(self.S[0], self.S[1]), mode='bilinear', align_corners=False))
        else:
            p = self.conv_layers_post(f)
            p = self.pred(p)

        return p


class MaskBranch(nn.Module):
    def __init__(self, indim=1024, outdim=256, conv_norm=""):
        super(MaskBranch, self).__init__()

        self.convR4 = Conv2d(indim+2, outdim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, outdim))
        self.ResR4 = ResBlock(outdim, outdim)
        self.RefR3 = Refine(int(indim/2), outdim)
        self.RefR2 = Refine(int(indim/4), outdim)
        self.convR1 = Conv2d(outdim, outdim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, outdim),
                             activation=nn.ReLU())
        self.pred = Conv2d(outdim, outdim, kernel_size=1, padding=0, stride=1, norm=get_norm(conv_norm, outdim))
        init_msra_fill(self)

    def forward(self, r4, r3, r2):
        p = self.ResR4(self.convR4(addCoord(r4)))
        p = self.RefR3(r3, p)
        p = self.RefR2(r2, p)
        p = self.convR1(F.relu(p))
        p = self.pred(p)

        return p


class CategoryPreBranch(nn.Module):
    def __init__(self, S, indim, outdim, conv_norm=""):
        super(CategoryPreBranch, self).__init__()

        self.S = S
        self.conv_layers = nn.Sequential()

        self.cat_pre_conv1 = Conv2d(indim, outdim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, outdim))
        for k in range(2):
            conv = Conv2d(
                outdim,
                outdim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=get_norm(conv_norm, outdim),
                activation=nn.ReLU(),
            )
            self.conv_layers.add_module("cat_pre_conv{}".format(k + 2), conv)

        self.cat_pre_conv4 =  Conv2d(outdim, outdim, kernel_size=1, padding=0, stride=1, norm=get_norm(conv_norm, outdim))
        init_msra_fill(self)

    def forward(self, f):
        p1 = self.cat_pre_conv1(F.interpolate(f, size=(self.S[0], self.S[1]), mode='bilinear', align_corners=False))
        p = self.conv_layers(F.relu(p1))
        p = self.cat_pre_conv4(p)

        return p, p1


class CategoryPostBranch(nn.Module):
    def __init__(self, C, dim=256, conv_norm=""):
        super(CategoryPostBranch, self).__init__()

        self.conv_layers = nn.Sequential()
        for k in range(3):
            conv = Conv2d(
                dim,
                dim,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=get_norm(conv_norm, dim),
                activation=nn.ReLU(),
            )
            self.conv_layers.add_module("cat_post_conv{}".format(k + 1), conv)

        self.pred = Conv2d(dim, C, kernel_size=3, padding=1, stride=1)
        init_msra_fill(self)

    def forward(self, f):
        p = self.conv_layers(f)
        p = self.pred(p)

        return p


class TrackingBranch(nn.Module):
    def __init__(self, dim, conv_norm=""):
        super(TrackingBranch, self).__init__()

        self.conv_layers_share = nn.Sequential()

        for k in range(2):
            conv = Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, dim),
                          activation=nn.ReLU())
            self.conv_layers_share.add_module("track_S_conv{}".format(k + 1), conv)

        self.track_Q_conv1 = Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, dim))
        self.track_Q_conv2 = Conv2d(dim, int(dim/2), kernel_size=1, padding=0, stride=1, norm=get_norm(conv_norm, int(dim/2)))
        self.track_R_conv1 = Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, dim))
        self.track_R_conv2 = Conv2d(dim, int(dim/2), kernel_size=1, padding=0, stride=1, norm=get_norm(conv_norm, int(dim/2)))
        init_msra_fill(self)

    def forward(self, f1, f2):
        q = self.conv_layers_share(F.relu(f1))
        q = self.track_Q_conv1(q)
        q = self.track_Q_conv2(q)
        q = q.view(-1, q.size()[1], q.size()[2] * q.size()[3]) # b, c/2, S1*S2
        q = torch.transpose(q, 1, 2)  # b, S1*S2, c/2

        if len(f2.size()) == 5:
            B, D, T, S1, S2 = f2.size()
            concat_f2 = torch.transpose(f2, 1, 2)
            concat_f2 = concat_f2.reshape(B*T, D, S1, S2)
            r = self.conv_layers_share(F.relu(concat_f2))
            r = self.track_R_conv1(r)
            r = self.track_R_conv2(r)
            r = r.view(-1, r.size()[1], r.size()[2] * r.size()[3])

            pred = torch.bmm(q.repeat(T,1,1), r)    # b*t, S1*S2, S1*S2
            pred = pred.view(B, T, S1*S2, S1*S2)

            return  pred

        else:
            r = self.conv_layers_share(F.relu(f2))
            r = self.track_R_conv1(r)
            r = self.track_R_conv2(r)
            r = r.view(-1, r.size()[1], r.size()[2] * r.size()[3])  # b, c/2, S1*S2

            pred = torch.bmm(q, r)  # b, S1*S2, S1*S2
            pred = pred.unsqueeze(1)  # b, 1, S1*S2, S1*S2

            return pred


class Memory(nn.Module):
    def __init__(self, dim, conv_norm=""):
        super(Memory, self).__init__()
        self.conv1 = Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, norm=get_norm(conv_norm, dim))
        init_msra_fill(self)

    def readPreFeature(self, tra, q_cat, m_cat):
        # tra: b, S1*S2, S1*S2
        # q_cat: b, dim, S1, S2
        # m_cat: b, dim, S1, S2
        B, C, S1, S2 = q_cat.size()

        sim = F.softmax(tra, dim=2)

        mc = self.conv1(m_cat).view(B, C, S1 * S2)  # b, dim, S1*S2
        mc = torch.transpose(mc, 1, 2)  # b, S1*S2, dim

        mem = torch.bmm(sim, mc)  # b, S1*S2, dim
        mem = torch.transpose(mem, 1, 2)  # b, dim, S1*S2
        mem = mem.view(B, C, S1, S2)

        mem_out = q_cat + mem

        return mem_out

    def readMemory(self, tras, q_cat, m_cat):
        # tras: B, T, S1*S2, S1*S2
        # q_cat: B, dim, S1, S2
        # m_cat: B, dim, T, S1, S2
        B, C, T, S1, S2 = m_cat.size()
        # sim = torch.sigmoid(tras)
        sim = torch.transpose(tras, 1, 2)  # B, S1*S2, T, S1*S2
        sim = sim.reshape(B, S1 * S2, T * S1 * S2)
        sim = F.softmax(sim, dim=2)

        mc = torch.transpose(m_cat, 1, 2)   # B, T, dim, S1, S2
        mc = mc.reshape(B*T, C, S1, S2)     # B*T, dim, S1, S2
        mc = self.conv1(mc)
        mc = mc.reshape(B, T, C, S1, S2)
        mc = torch.transpose(mc, 1, 2)      # B, dim, T, S1, S2

        # mc = self.conv1(m_cat[:, :, 0, :, :]).unsqueeze(2)  # B, dim, 1, S1, S2
        # for i in range(1, T):
        #     mc = torch.cat((mc, self.conv1(m_cat[:, :, i, :, :]).unsqueeze(2)), dim=2)
        mc = mc.reshape(B, C, T * S1 * S2)
        mc = torch.transpose(mc, 1, 2)  # B, T*S1*S2, dim

        mem = torch.bmm(sim, mc)  # B, S**2, dim
        mem = torch.transpose(mem, 1, 2).view(B, C, S1, S2)  # B, dim, S**2 -> B, dim, S, S

        mem_out = q_cat + mem

        return mem_out

    def forward(self, *args, **kwargs):
        if args[0].dim() == 3:
            return self.readPreFeature(*args, **kwargs)
        elif args[0].dim() == 4:
            return self.readMemory(*args, **kwargs)
        else:
            print('Wrong parameters!!!')
            exit()


class VISOLO(nn.Module):
    def __init__(self, backbone, backbone_features, S, C, indim=1024, outdim=256, conv_norm=""):
        super(VISOLO, self).__init__()

        self.backbone = backbone
        self.backbone_features = backbone_features
        self.S = S
        self.C = C
        self.conv_norm = conv_norm

        self.Category_branch_pre = CategoryPreBranch(S, indim, outdim, conv_norm)
        self.Category_branch_post = CategoryPostBranch(C, outdim, conv_norm)
        self.Kernel_branch = KernelBranch(S, indim, outdim, conv_norm)
        self.Mask_branch = MaskBranch(indim, outdim, conv_norm)
        self.Tracking_branch = TrackingBranch(outdim, conv_norm)
        self.Memory_module = Memory(outdim, conv_norm)

    def getModelResults(self, frames):

        B,_,N,_,_ = frames.size()
        Fs_cat = torch.cat((frames[:,:,0,:,:], frames[:,:,1,:,:], frames[:,:,2,:,:]), dim=0)

        features = self.backbone(Fs_cat)
        Fs_r2, Fs_r3, Fs_r4 = [features[f] for f in self.backbone_features]
        Fs_cat_f, Fs_f = self.Category_branch_pre(Fs_r4)
        Fs_kernel_f = self.Kernel_branch(Fs_r4, True)

        pred_tra_1 = self.Tracking_branch(Fs_f[:B], Fs_f[B:B*2])
        pred_tra_2 = self.Tracking_branch(Fs_f[:B], Fs_f[B*2:B*3])
        pred_tra = torch.cat((pred_tra_1, pred_tra_2), dim=1)
        f1_mem_out = self.Memory_module(torch.cat((pred_tra_1, pred_tra_2), dim=1), Fs_cat_f[:B],
                                        torch.cat((Fs_cat_f[B:B*2].unsqueeze(2), Fs_cat_f[B*2:B*3].unsqueeze(2)),
                                                  dim=2))
        f1_k_mem_out = self.Memory_module(torch.cat((pred_tra_1, pred_tra_2), dim=1), Fs_kernel_f[:B],
                                        torch.cat((Fs_kernel_f[B:B*2].unsqueeze(2), Fs_kernel_f[B*2:B*3].unsqueeze(2)),
                                                  dim=2))

        pred_tra_1 = self.Tracking_branch(Fs_f[B:B*2], Fs_f[:B])
        pred_tra_2 = self.Tracking_branch(Fs_f[B:B*2], Fs_f[B * 2:B * 3])
        pred_tra = torch.cat((pred_tra, pred_tra_2), dim=1)
        f2_mem_out = self.Memory_module(torch.cat((pred_tra_1, pred_tra_2), dim=1), Fs_cat_f[B:B*2],
                                        torch.cat((Fs_cat_f[:B].unsqueeze(2), Fs_cat_f[B * 2:B * 3].unsqueeze(2)),
                                                  dim=2))
        f2_k_mem_out = self.Memory_module(torch.cat((pred_tra_1, pred_tra_2), dim=1), Fs_kernel_f[B:B*2],
                                        torch.cat((Fs_kernel_f[:B].unsqueeze(2), Fs_kernel_f[B * 2:B * 3].unsqueeze(2)),
                                                  dim=2))

        pred_tra_1 = self.Tracking_branch(Fs_f[B * 2:B * 3], Fs_f[:B])
        pred_tra_2 = self.Tracking_branch(Fs_f[B * 2:B * 3], Fs_f[B:B * 2])
        f3_mem_out = self.Memory_module(torch.cat((pred_tra_1, pred_tra_2), dim=1), Fs_cat_f[B * 2:B * 3],
                                        torch.cat((Fs_cat_f[:B].unsqueeze(2), Fs_cat_f[B:B*2].unsqueeze(2)),
                                                  dim=2))
        f3_k_mem_out = self.Memory_module(torch.cat((pred_tra_1, pred_tra_2), dim=1), Fs_kernel_f[B * 2:B * 3],
                                        torch.cat((Fs_kernel_f[:B].unsqueeze(2), Fs_kernel_f[B:B*2].unsqueeze(2)),
                                                  dim=2))

        Fs_mem_out = torch.cat((f1_mem_out, f2_mem_out, f3_mem_out), dim=0)
        Fs_k_mem_out = torch.cat((f1_k_mem_out, f2_k_mem_out, f3_k_mem_out), dim=0)
        Fs_cat = self.Category_branch_post(Fs_mem_out)  # 3B, dim, S1, S2
        Fs_kernel = self.Kernel_branch(Fs_k_mem_out, False)
        Fs_mask = self.Mask_branch(Fs_r4, Fs_r3, Fs_r2)

        pred_masks = torch.cat(
            (Fs_mask[:B].unsqueeze(2), Fs_mask[B:B*2].unsqueeze(2), Fs_mask[B*2:B*3].unsqueeze(2)), dim=2)
        pred_kernels = torch.cat(
            (Fs_kernel[:B].unsqueeze(2), Fs_kernel[B:B*2].unsqueeze(2), Fs_kernel[B*2:B*3].unsqueeze(2)), dim=2)
        pred_cats = torch.cat(
            (Fs_cat[:B].unsqueeze(2), Fs_cat[B:B*2].unsqueeze(2), Fs_cat[B*2:B*3].unsqueeze(2)), dim=2)

        return pred_masks, pred_kernels, pred_cats, pred_tra

    def VISOLOTest(self, frame, m_frame_f, m_cat_f, m_kernel_f):
        # frames: B,C,H,W
        # m_frame_f: B, dim, T, S, S or B, dim, S, S
        # m_cat_f: B, dim, T, S, S or B, dim, S, S

        features = self.backbone(frame)
        F_r2, F_r3, F_r4 = [features[f] for f in self.backbone_features]
        cat_f, frame_f = self.Category_branch_pre(F_r4)
        kernel_f = self.Kernel_branch(F_r4, True)
        # pre_mem_kernel = None

        if m_frame_f is None:
            pred_tra = self.Tracking_branch(frame_f, frame_f)
            mem_out = self.Memory_module(pred_tra[:, 0, :, :], cat_f, cat_f)
            k_mem_out = self.Memory_module(pred_tra[:, 0, :, :], kernel_f, kernel_f)
        elif m_frame_f.dim() == 4:
            pred_tra = self.Tracking_branch(m_frame_f, frame_f)
            pred_tra_rv = self.Tracking_branch(frame_f, m_frame_f)
            mem_out = self.Memory_module(pred_tra_rv[:, 0, :, :], cat_f, m_cat_f)
            k_mem_out = self.Memory_module(pred_tra_rv[:, 0, :, :], kernel_f, m_kernel_f)

        else:
            pred_tra = self.Tracking_branch(m_frame_f[:, :, -1, :, :], frame_f)
            pred_tra_rv = self.Tracking_branch(frame_f, m_frame_f)
            mem_out = self.Memory_module(pred_tra_rv, cat_f, m_cat_f)
            k_mem_out = self.Memory_module(pred_tra_rv, kernel_f, m_kernel_f)

        frame_cat = self.Category_branch_post(mem_out)
        frame_kernel = self.Kernel_branch(k_mem_out, False)
        frame_mask = self.Mask_branch(F_r4, F_r3, F_r2)

        pred_masks = frame_mask.unsqueeze(2)
        pred_kernels = frame_kernel.unsqueeze(2)
        pred_cats = frame_cat.unsqueeze(2)

        if m_frame_f is None:
            return pred_masks.detach(), pred_kernels.detach(), torch.sigmoid(pred_cats.detach()), frame_f.detach(), \
                   cat_f.detach(), kernel_f.detach()
        else:
            return pred_masks.detach(), pred_kernels.detach(), torch.sigmoid(pred_cats.detach()), \
                   torch.sigmoid(pred_tra.detach()), frame_f.detach(), cat_f.detach(), kernel_f.detach()


    def getGridWeight(self, pre_frame_f, frame_f, post_frame_f):

        if pre_frame_f is None:
            ID_match = torch.sigmoid(self.Tracking_branch(frame_f, post_frame_f))   # B, 1, S**2, S**2 (B=1)
            grid_weight = torch.max(ID_match, dim=3)[0]    # B, 1, S**2

            return grid_weight

        elif post_frame_f is None:
            ID_match = torch.sigmoid(self.Tracking_branch(frame_f, pre_frame_f))  # B, 1, S**2, S**2 (B=1)
            grid_weight = torch.max(ID_match, dim=3)[0]  # B, 1, S**2

            return grid_weight

        else:
            pre_frames_f = torch.cat((pre_frame_f.unsqueeze(2), post_frame_f.unsqueeze(2)), dim=2)
            ID_match = torch.sigmoid(self.Tracking_branch(frame_f, pre_frames_f))   # B, 2, S1*S2, S1*S2
            grid_weight = torch.mean(torch.max(ID_match, dim=3)[0], 1, True)    # B, 1, S1*S2

            return grid_weight

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return self.getModelResults(*args, **kwargs)
        elif len(args) == 3:
            return self.getGridWeight(*args, **kwargs)
        elif len(args) == 4:
            return self.VISOLOTest(*args, **kwargs)
        else:
            print('Wrong parameters!!!')
            exit()


class SetCriterion(nn.Module):
    def __init__(self, FL_alpha, FL_gamma, DL_eps, mask_loss_weight):
        super(SetCriterion, self).__init__()
        self.category_loss = CategoryLoss(alpha=FL_alpha, gamma=FL_gamma)
        self.mask_loss = MaskLoss(eps=DL_eps, weight=mask_loss_weight)
        self.tracking_loss = TrackingLoss(alpha=FL_alpha, gamma=FL_gamma)

    def forward (self, pred_cats, pred_masks, pred_kernels, pred_tracking,
                 gt_grid_cats, gt_grid_masks, gt_grid_tracking):
        L_category = self.category_loss(pred_cats, gt_grid_cats)
        L_mask = self.mask_loss(pred_masks, pred_kernels, gt_grid_masks)
        L_tracking = self.tracking_loss(pred_tracking, gt_grid_tracking)

        return {
            "loss_cat": L_category,
            "loss_mask": L_mask,
            "loss_tracking": L_tracking,
        }


class DataUtils():
    def __init__(self, device, C, S=(12, 21), eps=0.2):
        self.C = C
        self.S = S
        self.eps = eps
        self.device = device

    def getMaskCenterAndSize(self, masks):
        # masks: len, H, W
        _, H, W = masks.size()

        ys = torch.arange(0, H, dtype=torch.float32, device=self.device)
        xs = torch.arange(0, W, dtype=torch.float32, device=self.device)

        n_valid = masks.sum(dim=(1,2)).clamp(min=1e-6)
        m_y = (masks * ys[:,None])
        m_x = (masks * xs)

        center_y = m_y.sum(dim=(1,2)) / n_valid
        center_x = m_x.sum(dim=(1,2)) / n_valid

        mask_h = m_y.amax(dim=(1,2)) - m_y.amin(dim=(1,2)) + 1
        mask_w = m_x.amax(dim=(1,2)) - m_x.amin(dim=(1,2)) + 1

        return center_y, center_x, mask_h, mask_w

    def getGridCenter(self, cell_h, cell_w, i, j):
        y = cell_h * i + cell_h / 2
        x = cell_w * j + cell_w / 2

        return torch.tensor(y), torch.tensor(x)

    def getGridClassAndMask(self, instances):
        # instances(Class Instances): the number of sampling frames

        N = len(instances)
        L = len(instances[0])
        H, W = instances[0].image_size
        S1, S2 = self.S

        grid_class = torch.zeros((self.C, N, S1, S2), dtype=torch.float32, device=self.device)
        grid_mask = torch.zeros((S1*S2, N, H, W), dtype=torch.float32, device=self.device)
        mask_center = torch.zeros((N, S1, S2, 3), dtype=torch.float32, device=self.device)
        grid_id = torch.zeros((N, L, S1*S2), dtype=torch.uint8, device=self.device)
        grid_id_matching = torch.zeros((N, S1*S2, S1*S2), dtype=torch.float32, device=self.device)

        cell_h = (H * 1.) / S1
        cell_w = (W * 1.) / S2

        for n in range(N):
            inst = instances[n].to(self.device)
            masks = inst.gt_masks.tensor    # L, H, W
            classes = inst.gt_classes   # L
            ids = inst.gt_ids   # L
            center_y, center_x, mask_h, mask_w = self.getMaskCenterAndSize(masks)
            delta_y = mask_h * self.eps
            delta_x = mask_w * self.eps
            for idx in range(L):
                 if torch.sum(masks[idx]) > 0:
                     grid_start_y = max(torch.div((center_y[idx] - delta_y[idx] / 2), cell_h, rounding_mode='floor'), 0)
                     grid_end_y = min(torch.div((center_y[idx] + delta_y[idx] / 2), cell_h, rounding_mode='floor'), S1 - 1)
                     grid_start_x = max(torch.div((center_x[idx] - delta_x[idx] / 2), cell_w, rounding_mode='floor'), 0)
                     grid_end_x = min(torch.div((center_x[idx] + delta_x[idx] / 2), cell_w, rounding_mode='floor'), S2 - 1)

                     for i in range(int(grid_start_y), int(grid_end_y) + 1):
                         for j in range(int(grid_start_x), int(grid_end_x) + 1):
                             if torch.sum(grid_class[:, n, i, j]) > 0:
                                 grid_center_y, grid_center_x = self.getGridCenter(cell_h, cell_w, i, j)
                                 m_y, m_x, o_idx = mask_center[n, i, j]
                                 o_idx = int(o_idx)
                                 d = (grid_center_y - m_y) ** 2 + (grid_center_x - m_x) ** 2
                                 new_d = (grid_center_y - center_y[idx]) ** 2 + (grid_center_x - center_x[idx]) ** 2
                                 if new_d < d:
                                     grid_class[:, n, i, j] = 0
                                     grid_mask[i * S2 + j, n, :, :] = 0
                                     grid_id[n, o_idx, i * S2 + j] = 0
                                 else:
                                     continue

                             grid_class[int(classes[idx]), n, i, j] = 1
                             grid_mask[i * S2 + j, n, :, :] = masks[idx]
                             mask_center[n, i, j] = torch.tensor([center_y[idx], center_x[idx], idx])
                             grid_id[n, idx, i * S2 + j] = 1

        for idx in range(L):
            ys_0 = torch.nonzero(grid_id[0, idx], as_tuple=True)[0]
            ys_1 = torch.nonzero(grid_id[1, idx], as_tuple=True)[0]
            ys_2 = torch.nonzero(grid_id[2, idx], as_tuple=True)[0]

            for y0 in ys_0:
                for y1 in ys_1:
                    grid_id_matching[0, y0, y1] = 1
                for y2 in ys_2:
                    grid_id_matching[1, y0, y2] = 1

            for y1 in ys_1:
                for y2 in ys_2:
                    grid_id_matching[2, y1, y2] = 1

        return grid_class, grid_mask, grid_id_matching

    def getGridGT(self, batch_dict):
        B = len(batch_dict)
        N = len(batch_dict[0]['instances'])
        L = len(batch_dict[0]['instances'][0])
        H, W = batch_dict[0]['instances'][0].image_size
        S1, S2 = self.S

        GT_classes = torch.zeros((B, self.C, N, S1, S2), dtype=torch.float32, device=self.device)
        GT_masks = torch.zeros((B, S1 * S2, N, H, W), dtype=torch.float32, device=self.device)
        GT_tracking = torch.zeros((B, N, S1 * S2, S1 * S2), dtype=torch.float32, device=self.device)

        for b in range(B):
            grid_class, grid_mask, grid_tracking = self.getGridClassAndMask(batch_dict[b]['instances'])
            GT_classes[b] = grid_class
            GT_masks[b] = grid_mask
            GT_tracking[b] = grid_tracking

        return GT_masks, GT_classes, GT_tracking







