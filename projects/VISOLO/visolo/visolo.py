import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy
import cv2
from scipy.ndimage.morphology import binary_dilation
from PIL import Image, ImageDraw, ImageFont
import os

from detectron2.modeling import META_ARCH_REGISTRY, build_backbone
from detectron2.structures import Boxes, ImageList, Instances

from .models.visolo_model import VISOLO, SetCriterion, DataUtils

__all__ = ["Visolo"]

@META_ARCH_REGISTRY.register()
class Visolo(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.num_frames = cfg.INPUT.SAMPLING_FRAME_NUM
        self.data_eps = cfg.INPUT.SAMPLING_EPS

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.num_class = cfg.MODEL.VISOLO.NUM_CLASSES

        self.mask_weight = cfg.MODEL.VISOLO.MASK_WEIGHT
        self.FL_alpha = cfg.MODEL.VISOLO.FOCAL_LOSS_ALPHA
        self.FL_gamma = cfg.MODEL.VISOLO.FOCAL_LOSS_GAMMA
        self.DL_eps = cfg.MODEL.VISOLO.DICE_LOSS_EPS

        self.S = cfg.MODEL.VISOLO.GRID_NUM
        self.indim = cfg.MODEL.VISOLO.INDIM
        self.outdim = cfg.MODEL.VISOLO.OUTDIM
        self.norm = cfg.MODEL.VISOLO.NORM

        self.tracking_thr = cfg.MODEL.VISOLO.TRACKING_THR
        self.score_thr = cfg.MODEL.VISOLO.SCORE_THR
        self.mask_thr = cfg.MODEL.VISOLO.MASK_THR
        self.update_thr = cfg.MODEL.VISOLO.UPDATE_THR
        self.kernel = cfg.MODEL.VISOLO.KERNEL
        self.sigma = cfg.MODEL.VISOLO.SIGMA
        self.nms_pre = cfg.MODEL.VISOLO.NMS_PRE

        backbone = build_backbone(cfg)
        backbone_features = cfg.MODEL.RESNETS.OUT_FEATURES
        self.model = VISOLO(backbone, backbone_features, self.S, self.num_class, self.indim, self.outdim, self.norm)
        self.criterion = SetCriterion(self.FL_alpha, self.FL_gamma, self.DL_eps, self.mask_weight)
        self.data_utils = DataUtils(self.device, self.num_class, self.S, self.data_eps)
        self.tracking_module = self.model.Tracking_branch

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def preprocess_image(self, batched_inputs, is_eval=False):
        B = len(batched_inputs)
        N = len(batched_inputs[0]['image'])
        dim, H, W = batched_inputs[0]['image'][0].size()

        frames = torch.zeros((B, dim, N, H, W), dtype=torch.float32, device=self.device)
        for b in range(B):
            for n in range(N):
                frames[b,:,n,:,:] = self.normalizer(batched_inputs[b]['image'][n].to(self.device))

        if is_eval:
            return frames, batched_inputs[0]['height'], batched_inputs[0]['width'], batched_inputs[0]['video_id']
        else:
            return frames

    def getCandidateDict(self, f_idx, valid_idx, tra_feature, idx_to_inst):
        candidate = {}
        candidate['f_idx'] = f_idx
        candidate['valid_idx'] = valid_idx
        candidate['tra_feature'] = tra_feature
        candidate['idx_mapping'] = idx_to_inst

        return candidate

    def forward(self, batched_inputs):

        if self.training:
            frames = self.preprocess_image(batched_inputs)
            pred_masks, pred_kernels, pred_cats, pred_tracking = self.model(frames)
            GT_masks, GT_classes, GT_tracking = self.data_utils.getGridGT(batched_inputs)

            loss_dict = self.criterion(pred_cats, pred_masks, pred_kernels, pred_tracking,
                                       GT_classes, GT_masks, GT_tracking)

            return loss_dict

        else:
            frames, v_h, v_w, v_id = self.preprocess_image(batched_inputs, is_eval=True)
            N = frames.size()[2]

            tra_candidates = []

            pred_masks_0, pred_kernel_0, pred_cats_0, frame_f, cat_f, kernel_f = self.model(frames[:, :, 0, :, :], None, None, None)
            m_frame_f = frame_f.unsqueeze(2)
            m_cat_f = cat_f.unsqueeze(2)
            m_kernel_f = kernel_f.unsqueeze(2)
            pred_masks_1, pred_kernel_1, pred_cats_1, pred_tracking, frame_f, cat_f, kernel_f = self.model(frames[:, :, 1, :, :], frame_f,
                                                                                            cat_f, kernel_f)

            grid_weight = None
            inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori, f0_result, valid_tra, valid_ind_map_to_inst, \
            _, _, tra_candidates, f0_valid_ind \
                = self.getTestResultV4(pred_masks_0[0], pred_kernel_0[0], pred_cats_0[0], pred_tracking[0],
                                            None,
                                            m_frame_f[:, :, -1, :, :], N, 0, [], [], [], [],
                                            None, None, [], tra_candidates)

            m_frame_f = torch.cat((m_frame_f, frame_f.unsqueeze(2)), dim=2)
            m_cat_f = torch.cat((m_cat_f, cat_f.unsqueeze(2)), dim=2)
            m_kernel_f = torch.cat((m_kernel_f, kernel_f.unsqueeze(2)), dim=2)

            pred_masks_2, pred_kernel_2, pred_cats_2, pred_tracking, frame_f, cat_f, kernel_f = self.model(frames[:, :, 2, :, :],
                                                                                            m_frame_f,
                                                                                            m_cat_f, m_kernel_f)

            grid_weight = self.model(m_frame_f[:, :, -2, :, :], m_frame_f[:, :, -1, :, :], None)
            inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori, f0_result, valid_tra, valid_ind_map_to_inst, \
            pre_inst_tra_check, pre_ind_map_inst, tra_candidates, f1_valid_ind \
                = self.getTestResultV4(pred_masks_1[0], pred_kernel_1[0], pred_cats_1[0], pred_tracking[0],
                                            grid_weight,
                                            m_frame_f[:, :, -1, :, :], N, 1, inst_masks, inst_cats, inst_cat_scores,
                                            inst_cat_scores_ori,
                                            f0_result, valid_tra, valid_ind_map_to_inst, tra_candidates)

            if pre_inst_tra_check is not None and (pre_inst_tra_check == 0).sum() > 0:
                pre_valid_ind = f0_valid_ind[pre_inst_tra_check == 0]
                tra_candidates.append(self.getCandidateDict(0, pre_valid_ind, m_frame_f[:, :, -2, :, :], pre_ind_map_inst))
            m_frame_f = torch.cat((m_frame_f, frame_f.unsqueeze(2)), dim=2)
            m_cat_f = torch.cat((m_cat_f, cat_f.unsqueeze(2)), dim=2)
            m_kernel_f = torch.cat((m_kernel_f, kernel_f.unsqueeze(2)), dim=2)

            pred_masks_1 = pred_masks_2.clone()
            pred_kernel_1 = pred_kernel_2.clone()
            pred_cats_1 = pred_cats_2.clone()

            if f1_valid_ind is not None:
                f0_valid_ind = f1_valid_ind.clone()
            else:
                f0_valid_ind = f1_valid_ind

            for n in range(3, N + 1):
                if n == N:
                    grid_weight = self.model(m_frame_f[:,:,-2,:,:], m_frame_f[:,:,-1,:,:], m_frame_f[:,:,-3,:,:])
                    inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori, f0_result, valid_tra, valid_ind_map_to_inst, \
                    pre_inst_tra_check, pre_ind_map_inst, tra_candidates, f1_valid_ind \
                        = self.getTestResultV4(pred_masks_1[0], pred_kernel_1[0], pred_cats_1[0], None, grid_weight,
                                                    m_frame_f[:,:,-1,:,:], N, n - 1, inst_masks, inst_cats, inst_cat_scores,
                                                    inst_cat_scores_ori,
                                                    f0_result, valid_tra, valid_ind_map_to_inst, tra_candidates)
                    continue

                pred_masks_2, pred_kernel_2, pred_cats_2, pred_tracking, frame_f, cat_f, kernel_f = self.model(frames[:, :, n, :, :], m_frame_f,
                                                                                           m_cat_f, m_kernel_f)  # B,S**2,1,H,W / B,C,1,S,S / B,1,S**2,S**2

                grid_weight = self.model(m_frame_f[:,:,-2,:,:], m_frame_f[:,:,-1,:,:], m_frame_f[:,:,-3,:,:])
                inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori, f0_result, valid_tra, valid_ind_map_to_inst, \
                pre_inst_tra_check, pre_ind_map_inst, tra_candidates, f1_valid_ind \
                    = self.getTestResultV4(pred_masks_1[0], pred_kernel_1[0], pred_cats_1[0], pred_tracking[0], grid_weight, m_frame_f[:,:,-1,:,:],
                                                N, n - 1, inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori,
                                                f0_result, valid_tra, valid_ind_map_to_inst, tra_candidates)

                if pre_inst_tra_check is not None and (pre_inst_tra_check == 0).sum() > 0:
                    pre_valid_ind = f0_valid_ind[pre_inst_tra_check == 0]
                    tra_candidates.append(self.getCandidateDict(n - 2, pre_valid_ind, m_frame_f[:,:,-2,:,:], pre_ind_map_inst))

                if n % 5 == 2:
                    m_frame_f = torch.cat((m_frame_f, frame_f.unsqueeze(2)), dim=2)
                    m_cat_f = torch.cat((m_cat_f, cat_f.unsqueeze(2)), dim=2)
                    m_kernel_f = torch.cat((m_kernel_f, kernel_f.unsqueeze(2)), dim=2)
                else:
                    m_frame_f = torch.cat((m_frame_f[:, :, :-2], m_frame_f[:, :, -1:], frame_f.unsqueeze(2)),
                                          dim=2)
                    m_cat_f = torch.cat((m_cat_f[:, :, :-2], m_cat_f[:, :, -1:], cat_f.unsqueeze(2)), dim=2)
                    m_kernel_f = torch.cat((m_kernel_f[:, :, :-2], m_kernel_f[:, :, -1:], kernel_f.unsqueeze(2)), dim=2)

                pred_masks_1 = pred_masks_2.clone()
                pred_kernel_1 = pred_kernel_2.clone()
                pred_cats_1 = pred_cats_2.clone()

                if f1_valid_ind is not None:
                    f0_valid_ind = f1_valid_ind.clone()
                else:
                    f0_valid_ind = f1_valid_ind

            if isinstance(inst_masks, list):
                return None

            inst_masks = F.interpolate(inst_masks, (v_h, v_w), mode='bilinear', align_corners=False)
            inst_masks = (inst_masks >= self.mask_thr).float()
            new_inst_cat_scores, new_inst_cats = self.getAverageCat(inst_cat_scores_ori)

            video_output = {
                "pred_scores": new_inst_cat_scores,
                "pred_labels": new_inst_cats,
                "pred_masks": inst_masks,
            }

            return video_output

    def getAverageCat(self, cat_scores_ori):
        # cat_scores_ori: K, v_l, 40
        K, L, _ = cat_scores_ori.size()
        valid_frame_num = torch.count_nonzero(torch.sum(cat_scores_ori, dim=2), dim=1).view(K, 1)
        avg_scores = torch.div(torch.sum(cat_scores_ori, dim=1), valid_frame_num.expand(K, self.num_class))
        cat_scores, cats = torch.max(avg_scores, dim=1)

        return cat_scores, cats

    def getTestResultV4(self, N_pred_masks, N_pred_kernels, N_pred_cats, N_pred_tra, grid_weight, tra_feature, N, f_idx,
                        inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori, f0_result=None,
                        valid_tra=None, valid_ind_map_to_inst=None, tra_candidates=None):

        # N_pred_masks : dim, 1, H/4, W/4
        # N_pred_kernels : dim, 1, S1, S2
        # N_pred_cats : C, 1, S1, S2
        # N_pred_tra : 1, S1*S2, S1*S2

        inst_masks = inst_masks  # K, N, H, W
        inst_cats = inst_cats  # K, N
        inst_cat_scores = inst_cat_scores  # K, N
        inst_cat_scores_ori = inst_cat_scores_ori  # K, N, 40
        f0_result = f0_result
        valid_tra = valid_tra
        valid_ind_map_to_inst = valid_ind_map_to_inst
        N = N
        f_idx = f_idx
        pre_inst_tra_check = None
        pre_ind_map_inst = None
        f0_valid_ind = None
        if grid_weight is not None:
            grid_weight = grid_weight[0,0]

        if f0_result is None:
            f0_result = self.getSegMaskV5(N_pred_masks[:,0,:,:], N_pred_kernels[:,0,:,:], N_pred_cats[:,0,:,:], grid_weight)

            if f0_result is not None:
                f0_seg_masks, f0_cat_labels, f0_cat_scores, f0_cat_scores_ori, f0_valid_ind = f0_result
                k0, _, _ = f0_seg_masks.size()
                if f_idx != N - 1:
                    valid_tra = N_pred_tra[0, f0_valid_ind, :]  # k0, S**2
                inst_num = len(inst_masks)
                no_match_ind = [x for x in range(k0)]
                inst_idx, tra_candidates = self.getTrackInfo(tra_candidates, tra_feature, no_match_ind,
                                                             valid_ind=f0_valid_ind)
                map_num = 0
                for i in range(k0):
                    if inst_idx[i] != -1:
                        if inst_masks[inst_idx[i]][f_idx].sum() != 0:
                            print('Error and mask in inst track!!!')
                            exit()
                        else:
                            inst_masks[inst_idx[i]][f_idx] = f0_seg_masks[i, :, :]
                            inst_cats[inst_idx[i]][f_idx] = f0_cat_labels[i]
                            inst_cat_scores[inst_idx[i]][f_idx] = f0_cat_scores[i]
                            inst_cat_scores_ori[inst_idx[i]][f_idx] = f0_cat_scores_ori[i]
                            valid_ind_map_to_inst.append(inst_idx[i])

                    else:
                        _, H, W = f0_seg_masks.size()
                        masks = torch.zeros((1, N, H, W), device=self.device)
                        cats = torch.full((1, N), -1, device=self.device)
                        cat_scores = torch.zeros((1, N), device=self.device)
                        cat_scores_ori = torch.zeros((1, N, self.num_class), device=self.device)

                        masks[0, f_idx] = f0_seg_masks[i, :, :]
                        cats[0, f_idx] = f0_cat_labels[i]
                        cat_scores[0, f_idx] = f0_cat_scores[i]
                        cat_scores_ori[0, f_idx] = f0_cat_scores_ori[i]

                        if isinstance(inst_masks, list):
                            inst_masks = masks
                            inst_cats = cats
                            inst_cat_scores = cat_scores
                            inst_cat_scores_ori = cat_scores_ori
                        else:
                            inst_masks = torch.cat((inst_masks, masks), dim=0)
                            inst_cats = torch.cat((inst_cats, cats), dim=0)
                            inst_cat_scores = torch.cat((inst_cat_scores, cat_scores), dim=0)
                            inst_cat_scores_ori = torch.cat((inst_cat_scores_ori, cat_scores_ori), dim=0)

                        valid_ind_map_to_inst.append(inst_num + map_num)
                        map_num+=1

        else:
            f0_result = self.getSegMaskV5(N_pred_masks[:,0,:,:], N_pred_kernels[:,0,:,:], N_pred_cats[:,0,:,:], grid_weight)

            if f0_result is not None:
                f0_seg_masks, f0_cat_labels, f0_cat_scores, f0_cat_scores_ori, f0_valid_ind = f0_result
                k1, _, _ = f0_seg_masks.size()
                no_match_ind = []
                temp_map_ind = [0 for _ in range(k1)]
                pre_inst_tra_check = torch.zeros((valid_tra.size()[0]))
                inst_num = len(inst_masks)

                valid_tra = valid_tra[:, f0_valid_ind]  # k0, k1

                for i in range(k1):
                    tra_sort_ind = torch.argsort(valid_tra[:, i], descending=True)
                    check_match = 0

                    for ind in tra_sort_ind:
                        inst_map_ind = valid_ind_map_to_inst[int(ind)]
                        if inst_masks[inst_map_ind][f_idx].sum() == 0 and valid_tra[int(ind), i] >= self.tracking_thr:
                            inst_masks[inst_map_ind][f_idx] = f0_seg_masks[i, :, :]
                            inst_cats[inst_map_ind][f_idx] = f0_cat_labels[i]
                            inst_cat_scores[inst_map_ind][f_idx] = f0_cat_scores[i]
                            inst_cat_scores_ori[inst_map_ind][f_idx] = f0_cat_scores_ori[i]
                            check_match = 1
                            temp_map_ind[i] = inst_map_ind
                            pre_inst_tra_check[int(ind)] = 1
                            break

                    if check_match == 0:
                        no_match_ind.append(i)

                valid_ind = f0_valid_ind[no_match_ind]
                inst_idx, tra_candidates = self.getTrackInfo(tra_candidates, tra_feature, no_match_ind,
                                                             valid_ind=valid_ind)
                map_num = 0
                for i in range(len(no_match_ind)):
                    ind = no_match_ind[i]
                    if inst_idx[i] != -1:
                        if inst_masks[inst_idx[i]][f_idx].sum() != 0:
                            print('Error add mask in inst track!!!')
                            exit()
                        else:
                            inst_masks[inst_idx[i]][f_idx] = f0_seg_masks[ind, :, :]
                            inst_cats[inst_idx[i]][f_idx] = f0_cat_labels[ind]
                            inst_cat_scores[inst_idx[i]][f_idx] = f0_cat_scores[ind]
                            inst_cat_scores_ori[inst_idx[i]][f_idx] = f0_cat_scores_ori[ind]
                            temp_map_ind[ind] = inst_idx[i]

                    else:
                        _, H, W = f0_seg_masks.size()
                        masks = torch.zeros((1, N, H, W), device=self.device)
                        cats = torch.full((1, N), -1, device=self.device)
                        cat_scores = torch.zeros((1, N), device=self.device)
                        cat_scores_ori = torch.zeros((1, N, self.num_class), device=self.device)

                        masks[0, f_idx] = f0_seg_masks[ind, :, :]
                        cats[0, f_idx] = f0_cat_labels[ind]
                        cat_scores[0, f_idx] = f0_cat_scores[ind]
                        cat_scores_ori[0, f_idx] = f0_cat_scores_ori[ind]

                        inst_masks = torch.cat((inst_masks, masks), dim=0)
                        inst_cats = torch.cat((inst_cats, cats), dim=0)
                        inst_cat_scores = torch.cat((inst_cat_scores, cat_scores), dim=0)
                        inst_cat_scores_ori = torch.cat((inst_cat_scores_ori, cat_scores_ori), dim=0)

                        temp_map_ind[ind] = inst_num + map_num
                        map_num += 1

                pre_ind_map_inst = [k for idx, k in enumerate(valid_ind_map_to_inst)
                                    if pre_inst_tra_check[idx] == 0]

                valid_ind_map_to_inst = temp_map_ind
                if f_idx != N - 1:
                    valid_tra = N_pred_tra[0, f0_valid_ind, :]  # k1, S**2

            else:
                pre_inst_tra_check = torch.zeros((valid_tra.size()[0]))
                pre_ind_map_inst = copy.deepcopy(valid_ind_map_to_inst)
                valid_tra = None
                valid_ind_map_to_inst = []

        return inst_masks, inst_cats, inst_cat_scores, inst_cat_scores_ori,\
               f0_result, valid_tra, valid_ind_map_to_inst,\
               pre_inst_tra_check, pre_ind_map_inst, tra_candidates, f0_valid_ind

    def getSegMaskV5(self, pred_masks, pred_kernels, pred_cats, grid_weight):
        # pred_masks : dim, H/4, W/4
        # pred_kernels : dim, S1, S2
        # pred_cats : C, S1, S2
        # grid_weight : S1*S2

        _, H, W = pred_masks.size()
        C, S1, S2 = pred_cats.size()

        cat_scores = pred_cats.reshape(-1, S1 * S2).transpose(1, 0)  # S**2, C
        cat_scores_ori = cat_scores.clone()
        cat_scores, cat_labels = cat_scores.max(1)  # S**2
        if grid_weight is not None:
            cat_scores *= grid_weight
        cat_scores[cat_scores < self.score_thr] = 0.
        valid_ind = cat_scores.nonzero()[:, 0]

        if valid_ind.sum() == 0:
            return None

        kernels = pred_kernels.reshape(-1, S1*S2).transpose(1, 0)   # S1*S2, dim
        kernels = kernels[valid_ind]

        seg_preds = self.getMaskMap(pred_masks, kernels)
        seg_masks = (seg_preds > self.mask_thr).float()
        cat_scores = cat_scores[valid_ind]
        cat_labels = cat_labels[valid_ind]

        sum_masks = seg_masks.sum((1, 2)).float()
        seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
        cat_scores *= seg_scores

        sort_ind = torch.argsort(cat_scores, descending=True)
        if sort_ind.size()[0] > self.nms_pre:
            sort_ind = sort_ind[:self.nms_pre]
        seg_masks = seg_masks[sort_ind, :, :]
        cat_scores = cat_scores[sort_ind]
        cat_labels = cat_labels[sort_ind]
        valid_ind = valid_ind[sort_ind]

        cat_scores = self.matrix_nms(seg_masks, cat_labels, cat_scores)

        keep = cat_scores >= self.update_thr
        if keep.sum() == 0:
            return None
        seg_masks = seg_masks[keep, :, :]
        cat_scores = cat_scores[keep]
        cat_labels = cat_labels[keep]
        valid_ind = valid_ind[keep]

        sort_ind = torch.argsort(cat_scores, descending=True)
        if sort_ind.size()[0] > 100:
            sort_ind = sort_ind[:100]
        seg_masks = seg_masks[sort_ind, :, :]
        cat_scores = cat_scores[sort_ind]
        cat_labels = cat_labels[sort_ind]
        valid_ind = valid_ind[sort_ind]

        cat_scores_ori = cat_scores_ori[valid_ind, :]

        for i in range(len(valid_ind) - 1):
            if seg_masks[i].sum() == 0:
                continue
            for j in range(i + 1, len(valid_ind)):
                inter_region = (seg_masks[i] * seg_masks[j]).sum()
                mask_region = seg_masks[j].sum()
                if inter_region / mask_region > 0.5:
                    seg_masks[j] = 0

        final_valid_ind = (seg_masks.sum((1, 2)) > 0)
        seg_masks = seg_masks[final_valid_ind, :, :]
        cat_scores = cat_scores[final_valid_ind]
        cat_labels = cat_labels[final_valid_ind]
        cat_scores_ori = cat_scores_ori[final_valid_ind, :]
        valid_ind = valid_ind[final_valid_ind]

        return seg_masks, cat_labels, cat_scores, cat_scores_ori, valid_ind

    def matrix_nms(self, seg_masks, cate_labels, cate_scores, sum_masks=None):
        """Matrix NMS for multi-class masks.

        Args:
            seg_masks (Tensor): shape (n, h, w)
            cate_labels (Tensor): shape (n), mask labels in descending order
            cate_scores (Tensor): shape (n), mask scores in descending order
            self.kernel (str):  'linear' or 'gauss'
            self.sigma (float): std in gaussian method
            sum_masks (Tensor): The sum of seg_masks

        Returns:
            Tensor: cate_scores_update, tensors of shape (n)
        """
        n_samples = len(cate_labels)
        if n_samples == 0:
            return []
        if sum_masks is None:
            sum_masks = seg_masks.sum((1, 2)).float()
        seg_masks = seg_masks.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0))
        label_matrix = label_matrix.float().triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

        # IoU decay
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = torch.exp(-1 * self.sigma * (decay_iou ** 2))
            compensate_matrix = torch.exp(-1 * self.sigma * (compensate_iou ** 2))
            decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        elif self.kernel == 'linear':
            decay_matrix = (1 - decay_iou) / (1 - compensate_iou)
            decay_coefficient, _ = decay_matrix.min(0)
        else:
            raise NotImplementedError

        # update the score.
        cate_scores_update = cate_scores * decay_coefficient
        return cate_scores_update

    def getTrackInfo(self, tra_candidates, tra_feature, idxs, valid_ind=None):
        can_num = len(tra_candidates)
        inst_idx = [-1 for _ in range(len(idxs))]
        if can_num == 0:
            return inst_idx, tra_candidates

        [k1] = valid_ind.size()
        tra_candidate_check = []
        for c in range(can_num-1, -1, -1):
            tra_check = []
            tra_candidate = tra_candidates[c]

            [k0] = tra_candidate['valid_idx'].size()
            pred_tra = torch.sigmoid(self.tracking_module(tra_candidate['tra_feature'], tra_feature))
            pred_tra = pred_tra[0,0]

            valid_tra = pred_tra[tra_candidate['valid_idx'],:]
            valid_tra = valid_tra[:, valid_ind]
            for i in range(k1):
                if inst_idx[i] == -1:
                    tra_sort_ind = torch.argsort(valid_tra[:, i], descending=True)
                    for ind in tra_sort_ind:
                        if valid_tra[int(ind), i] >= self.tracking_thr and int(ind) not in tra_check:
                            inst_idx[i] = tra_candidate['idx_mapping'][int(ind)]
                            tra_check.append(int(ind))
                            break

            valid_masks_idx = [x for x in range(k0) if x not in tra_check]

            tra_candidate['valid_idx'] = tra_candidate['valid_idx'][valid_masks_idx]
            tra_candidate['idx_mapping'] = [k for idx, k in enumerate(tra_candidate['idx_mapping'])
                                            if idx not in tra_check]
            if len(tra_candidate['idx_mapping']) == 0:
                tra_candidate_check.append(c)
            if -1 not in inst_idx:
                break

        tra_candidates = [k for idx, k in enumerate(tra_candidates)
                         if idx not in tra_candidate_check]

        return inst_idx, tra_candidates

    def getMaskMap(self, mask, kernel):
        # mask: dim, H/4, W/4
        # kernel: valid_idxs, dim
        # out_device = mask.device

        if not mask.is_cuda:
            mask = mask.to('cuda')
            kernel = kernel.to('cuda')
        num_kernel, _ = kernel.size()
        dim, H, W = mask.size()

        mask = mask.unsqueeze(0)  # 1, dim, H/4, W/4
        mask_map = F.conv2d(mask, kernel.view(num_kernel, dim, 1, 1))

        mask_map = F.interpolate(mask_map, scale_factor=4, mode='bilinear', align_corners=False).squeeze(0)
        mask_map = torch.sigmoid(mask_map)

        return mask_map



