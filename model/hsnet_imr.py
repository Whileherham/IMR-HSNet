r""" Hypercorrelation Squeeze Network """
import pdb
from functools import reduce
from operator import add

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .learner import HPNLearner


class HypercorrSqueezeNetwork_imr(nn.Module):
    # 与不是res的结构相比，这里就是让logit——mask和cam cat一下，然后过个卷积出结果，来显式地再用一下cam
    def __init__(self, backbone, use_original_imgsize):
        super(HypercorrSqueezeNetwork_imr, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=False)
            ckpt = torch.load('../Datasets_HSN/Pretrain/vgg16-397923af.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=False)
            ckpt = torch.load('../Datasets_HSN/Pretrain/resnet50-19c8e357.pth')
            self.backbone.load_state_dict(ckpt)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
            self.conv1024_512 = nn.Conv2d(1024, 512, kernel_size=1)

        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        # IMR
        self.state = nn.Parameter(torch.zeros([1, 128, 50, 50]))
        self.convz0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convz1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convz2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        self.convr0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convr1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convr2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        self.convh0 = nn.Conv2d(769, 512, kernel_size=1, padding=0)
        self.convh1 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)
        self.convh2 = nn.Conv2d(256, 64, kernel_size=3, padding=1, groups=8, dilation=1)

        # copied from hsnet-learner
        outch1, outch2, outch3 = 16, 64, 128
        self.decoder1 = nn.Sequential(nn.Conv2d(outch3, outch3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch3, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU())

        self.decoder2 = nn.Sequential(nn.Conv2d(outch2, outch2, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(outch2, 2, (3, 3), padding=(1, 1), bias=True))

        self.res = nn.Sequential(nn.Conv2d(3, 10, kernel_size=1),
                                 nn.GELU(),
                                 nn.Conv2d(10, 2, kernel_size=1))

    def forward(self, query_img, support_img, support_cam, query_cam,
                query_mask=None, support_mask=None, stage=2, w='same'):
        with torch.no_grad():
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats = self.extract_feats(
                support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)

            # extracting feature
            if len(query_feats) == 7:
                isvgg = True  # VGG
                q_mid_feat = F.interpolate(query_feats[3] + query_feats[4] + query_feats[5],
                                           (50, 50), mode='bilinear', align_corners=True)
                s_mid_feat = F.interpolate(support_feats[3] + support_feats[4] + support_feats[5],
                                           (50, 50), mode='bilinear', align_corners=True)
            else:
                isvgg = False  # R50
                q_mid_feat = F.interpolate(
                    query_feats[4] + query_feats[5] + query_feats[6] + query_feats[7] + query_feats[8] + query_feats[9],
                    (50, 50), mode='bilinear', align_corners=True)

                s_mid_feat = F.interpolate(
                    support_feats[4] + support_feats[5] + support_feats[6] + support_feats[7] + support_feats[8] +
                    support_feats[9],
                    (50, 50), mode='bilinear', align_corners=True)

            query_feats_masked = self.mask_feature(query_feats, support_cam.clone())
            support_feats_masked = self.mask_feature(support_feats, query_cam.clone())

            corr_query = Correlation.multilayer_correlation(query_feats, support_feats_masked, self.stack_ids)
            corr_support = Correlation.multilayer_correlation(support_feats, query_feats_masked, self.stack_ids)

            query_cam = query_cam.unsqueeze(1)
            support_cam = support_cam.unsqueeze(1)

        if not isvgg:
            # make feat dim in R50 same as VGG
            q_mid_feat = self.conv1024_512(q_mid_feat)
            s_mid_feat = self.conv1024_512(s_mid_feat)

        bsz = query_img.shape[0]
        state_query = self.state.expand(bsz, -1, -1, -1)
        state_support = self.state.expand(bsz, -1, -1, -1)

        losses = 0
        for ss in range(stage):
            # query
            after4d_query = self.hpn_learner.forward_conv4d(corr_query)
            imr_x_query = torch.cat([query_cam, after4d_query, q_mid_feat, state_query], dim=1)

            imr_x_query_z = self.convz0(imr_x_query)
            imr_z_query1 = self.convz1(imr_x_query_z[:, :256])
            imr_z_query2 = self.convz2(imr_x_query_z[:, 256:])
            imr_z_query = torch.sigmoid(torch.cat([imr_z_query1, imr_z_query2], dim=1))

            imr_x_query_r = self.convr0(imr_x_query)
            imr_r_query1 = self.convr1(imr_x_query_r[:, :256])
            imr_r_query2 = self.convr2(imr_x_query_r[:, 256:])
            imr_r_query = torch.sigmoid(torch.cat([imr_r_query1, imr_r_query2], dim=1))

            imr_x_query_h = self.convh0(
                torch.cat([query_cam, after4d_query, q_mid_feat, imr_r_query * state_query], dim=1))
            imr_h_query1 = self.convh1(imr_x_query_h[:, :256])
            imr_h_query2 = self.convh2(imr_x_query_h[:, 256:])
            imr_h_query = torch.cat([imr_h_query1, imr_h_query2], dim=1)

            state_new_query = torch.tanh(imr_h_query)
            state_query = (1 - imr_z_query) * state_query + imr_z_query * state_new_query

            # support
            after4d_support = self.hpn_learner.forward_conv4d(corr_support)
            imr_x_support = torch.cat([support_cam, after4d_support, s_mid_feat, state_support], dim=1)

            imr_x_support_z = self.convz0(imr_x_support)
            imr_z_support1 = self.convz1(imr_x_support_z[:, :256])
            imr_z_support2 = self.convz2(imr_x_support_z[:, 256:])
            imr_z_support = torch.sigmoid(torch.cat([imr_z_support1, imr_z_support2], dim=1))

            imr_x_support_r = self.convr0(imr_x_support)
            imr_r_support1 = self.convr1(imr_x_support_r[:, :256])
            imr_r_support2 = self.convr2(imr_x_support_r[:, 256:])
            imr_r_support = torch.sigmoid(torch.cat([imr_r_support1, imr_r_support2], dim=1))

            imr_x_support_h = self.convh0(
                torch.cat([support_cam, after4d_support, s_mid_feat, imr_r_support * state_support], dim=1))
            imr_h_support1 = self.convh1(imr_x_support_h[:, :256])
            imr_h_support2 = self.convh2(imr_x_support_h[:, 256:])
            imr_h_support = torch.cat([imr_h_support1, imr_h_support2], dim=1)

            state_new_support = torch.tanh(imr_h_support)
            state_support = (1 - imr_z_support) * state_support + imr_z_support * state_new_support

            # decoder
            hypercorr_decoded_s = self.decoder1(state_support + after4d_support)
            upsample_size = (hypercorr_decoded_s.size(-1) * 2,) * 2
            hypercorr_decoded_s = F.interpolate(hypercorr_decoded_s, upsample_size, mode='bilinear', align_corners=True)
            logit_mask_support = self.decoder2(hypercorr_decoded_s)

            hypercorr_decoded_q = self.decoder1(state_query + after4d_query)
            upsample_size = (hypercorr_decoded_q.size(-1) * 2,) * 2
            hypercorr_decoded_q = F.interpolate(hypercorr_decoded_q, upsample_size, mode='bilinear', align_corners=True)
            logit_mask_query = self.decoder2(hypercorr_decoded_q)

            logit_mask_support = self.res(
                torch.cat(
                    [logit_mask_support, F.interpolate(support_cam, (100, 100), mode='bilinear', align_corners=True)],
                    dim=1))
            logit_mask_query = self.res(
                torch.cat([logit_mask_query, F.interpolate(query_cam, (100, 100), mode='bilinear', align_corners=True)],
                          dim=1))

            # loss
            if query_mask is not None:  # for training
                if not self.use_original_imgsize:
                    logit_mask_query_temp = F.interpolate(logit_mask_query, support_img.size()[2:], mode='bilinear',
                                                          align_corners=True)
                    logit_mask_support_temp = F.interpolate(logit_mask_support, support_img.size()[2:], mode='bilinear',
                                                            align_corners=True)
                loss_q_stage = self.compute_objective(logit_mask_query_temp, query_mask)
                loss_s_stage = self.compute_objective(logit_mask_support_temp, support_mask)
                losses = losses + loss_q_stage + loss_s_stage

            if ss != stage - 1:
                support_cam = logit_mask_support.softmax(dim=1)[:, 1]
                query_cam = logit_mask_query.softmax(dim=1)[:, 1]
                query_feats_masked = self.mask_feature(query_feats, query_cam)
                support_feats_masked = self.mask_feature(support_feats, support_cam)
                corr_query = Correlation.multilayer_correlation(query_feats, support_feats_masked, self.stack_ids)
                corr_support = Correlation.multilayer_correlation(support_feats, query_feats_masked, self.stack_ids)

                query_cam = F.interpolate(query_cam.unsqueeze(1), (50, 50), mode='bilinear', align_corners=True)
                support_cam = F.interpolate(support_cam.unsqueeze(1), (50, 50), mode='bilinear', align_corners=True)

        if query_mask is not None:
            return logit_mask_query_temp, logit_mask_support_temp, losses
        else:
            # test
            if not self.use_original_imgsize:
                logit_mask_query = F.interpolate(
                    logit_mask_query, support_img.size()[2:], mode='bilinear', align_corners=True)
                logit_mask_support = F.interpolate(
                    logit_mask_support, support_img.size()[2:], mode='bilinear', align_corners=True)
            return logit_mask_query, logit_mask_support

    def mask_feature(self, features, support_mask):
        for idx, feature in enumerate(features):
            mask = F.interpolate(
                support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            features[idx] = features[idx] * mask
        return features

    def predict_mask_nshot(self, batch, nshot, stage):
        # Perform multiple prediction given (nshot) number of different support sets
        logit_mask_agg = 0
        for s_idx in range(nshot):
            logit_mask, logit_mask_s = self(query_img=batch['query_img'],
                                            support_img=batch['support_imgs'][:, s_idx],
                                            support_cam=batch['support_cams'][:, s_idx],
                                            query_cam=batch['query_cam'], stage=stage)
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)

            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1:
                return logit_mask_agg

        # Average & quantize predictions given threshold (=0.5)
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1

        return pred_mask

    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging
