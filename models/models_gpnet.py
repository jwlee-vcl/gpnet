from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import models.models_pointnet as pointnet
import models.gpnet_utils as gpnet_utils
import models.networks_gpnet as networks_gpnet

imagenet_stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}

class ResnetBase(nn.Module):
    def __init__(self, n_in_layers, n_out_classes, pretrained):
        super(ResnetBase, self).__init__()
        self.pretrained = pretrained
        resnet = models.resnet50(pretrained=self.pretrained)        
        if n_in_layers != 3:  # Number of input channels
            self.conv1 = nn.Conv2d(in_channels=n_in_layers, out_channels=64,
                                kernel_size=(7, 7), stride=(2, 2),
                                padding=(3, 3), bias=False)
        else:
            self.conv1 = resnet.conv1 # H/2

        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool # H/4

        # encoder
        self.layer1 = resnet.layer1 # H/4
        self.layer2 = resnet.layer2 # H/8
        self.layer3 = resnet.layer3 # H/16
        self.layer4 = resnet.layer4 # H/32

        self.avgpool = resnet.avgpool
        
        self.fc = nn.Linear(2048, n_out_classes)

    def forward(self, x):
        # encoder
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)        
        x = self.layer2(x)        
        x = self.layer3(x)        
        conv_output = x
        x = self.layer4(x) 
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x), conv_output, 

class GPNet_Vert(nn.Module):
    def __init__(self, opt):
        super(GPNet_Vert, self).__init__()

        self.n_vert_lines = opt.n_vert_lines
        self.n_vert_pts   = opt.n_vert_pts

        self.n_inlier_zvps = opt.n_inlier_zvps
        
        self.pointfeat = pointnet.PointNetBase(ich=opt.ich_point, 
                                                 och_inst=opt.ch_inst, och_global=opt.ch_global)
        self.linefeat  = pointnet.PointNetBase(ich=opt.ich_line, 
                                                 och_inst=opt.ch_inst, och_global=opt.ch_global)

        self.pointprob = pointnet.PointNetSeg2(ich_inst=opt.ch_inst, ich_global=opt.ch_global, och=1)
        self.lineprob  = pointnet.PointNetSeg2(ich_inst=opt.ch_inst, ich_global=opt.ch_global, och=1)

        self.sigmoid = torch.nn.Sigmoid()
                                    
    def forward(self, segs, out_dict=None):
        if out_dict is None:
            out_dict = {}
        
        with torch.no_grad():
            vsegs = gpnet_utils.init_filter_vert_segs(segs, n_lines=self.n_vert_lines)
            out_dict['vert/segs'] = vsegs

            vlines = gpnet_utils.segs2lines(vsegs, focal=1.0)
            out_dict['vert/lines'] = vlines

            vpts = gpnet_utils.generate_inter_pts(vlines, vlines, n_pts=self.n_vert_pts)
            out_dict['vert/pts'] = vpts

            vpts_n = F.normalize(vpts, p=2, dim=-1)          
            vpts_n = vpts_n.transpose(2, 1).contiguous()

            vlines_n = F.normalize(vlines, p=2, dim=-1)      
            vlines_n = vlines_n.transpose(2, 1).contiguous()

        pts_inst_feat, pts_global_feat = self.pointfeat(vpts_n)
        lines_inst_feat, lines_global_feat = self.linefeat(vlines_n)

        pts_prob   = self.pointprob(pts_inst_feat, lines_global_feat)
        out_dict['vert/pts_prob'] = pts_prob

        pts_weight = F.softmax(pts_prob, dim=1)
        
        zvp = torch.sum(pts_weight*vpts, dim=1)        
        out_dict['zvp'] = zvp

        # pts_prob = self.sigmoid(pts_prob)
        pts_prob = self.sigmoid(pts_weight)
        pts_prob = pts_prob.squeeze(-1)
        zvps = gpnet_utils.pick_hyps(vpts, pts_prob, k=self.n_inlier_zvps)
        out_dict['zvps'] = zvps

        return out_dict

class FrameSampler(nn.Module):
    def __init__(self, opt):
        super(FrameSampler, self).__init__()

        self.n_in_vert_lines = opt.n_in_vert_lines
        self.n_in_hori_lines = opt.n_in_hori_lines

        self.n_juncs = opt.n_juncs

        self.n_hori_lines = opt.n_hori_lines
        self.n_hori_pts = opt.n_hori_pts
        
        self.n_frames = opt.n_frames

    def forward(self, segs, zvp, zvps, seg_map, seg_mask, out_dict=None, in_dict=None):
        if out_dict is None:
            out_dict = {}
                        
        # junction filtering
        with torch.no_grad():
            (vsegs, hsegs) = (gpnet_utils.init_filter_segs(segs, zvp, 
                                 n_vert_lines=self.n_in_vert_lines, 
                                 n_hori_lines=self.n_in_hori_lines))
            out_dict['vsegs'] = vsegs
            out_dict['hsegs'] = hsegs
         
            (juncs, juncs_vsegs, juncs_vlines, juncs_hsegs, juncs_hlines) = (
                gpnet_utils.junc_filtering(vsegs, hsegs, n_juncs=self.n_juncs))
            out_dict['juncs'] = juncs
            out_dict['juncs_vsegs'] = juncs_vsegs
            out_dict['juncs_hsegs'] = juncs_hsegs
            out_dict['juncs_vlines'] = juncs_vlines
            out_dict['juncs_hlines'] = juncs_hlines

            (hsegs_p, hsegs_m) = (
                gpnet_utils.filter_lines_by_dir(juncs_hsegs, juncs_hlines, zvp, n_lines=self.n_hori_lines))
            out_dict['hsegs_p'] = hsegs_p
            out_dict['hsegs_m'] = hsegs_m
            
            hlines_p = gpnet_utils.segs2lines(hsegs_p, focal=1.0)
            hlines_m = gpnet_utils.segs2lines(hsegs_m, focal=1.0)
            out_dict['hlines_p'] = hlines_p
            out_dict['hlines_m'] = hlines_m

            hpts_p = gpnet_utils.generate_inter_pts(hlines_p, hlines_p, n_pts=self.n_hori_pts)
            hpts_m = gpnet_utils.generate_inter_pts(hlines_m, hlines_m, n_pts=self.n_hori_pts)
            out_dict['hpts_p'] = hpts_p
            out_dict['hpts_m'] = hpts_m
            
            (frames, focals) = gpnet_utils.generate_frames(zvps, hpts_p, hpts_m, n_frames=self.n_frames)
            out_dict['frames'] = frames # [b,n,9]
            out_dict['focals'] = focals # [b,n,1]

            act_maps, _, _ = gpnet_utils.generate_active_map(frames, focals, seg_map, seg_mask)
            out_dict['act_maps'] = act_maps

        return out_dict

class ScoreNet(nn.Module):
    def __init__(self, opt):
        super(ScoreNet, self).__init__()
        self.opt = opt
        self.resnet_new = ResnetBase(n_in_layers=(3 + 1 + 3 + 9 + 1), n_out_classes=1, pretrained=True)
        
    def forward(self, img, seg_map, seg_mask, frames, focals, out_dict=None, in_dict=None):
        '''
        img [b,c,h,w] 
        frames [b,n,9] 
        focals [b,n,1] 
        seg_map [b,h,w,4]
        seg_mask [b,h,w,1]
        '''
        if out_dict is None:
            out_dict = {}

        # normalize image
        img_r = (img[:,0:1,:,:] - imagenet_stats['mean'][0])/imagenet_stats['std'][0]
        img_g = (img[:,1:2,:,:] - imagenet_stats['mean'][1])/imagenet_stats['std'][1]
        img_b = (img[:,2:3,:,:] - imagenet_stats['mean'][2])/imagenet_stats['std'][2]
        img = torch.cat((img_r, img_g, img_b), dim=1)

        with torch.no_grad(): 
            act_maps, _, _ = gpnet_utils.generate_active_map(frames, focals, seg_map, seg_mask)
            out_dict['act_maps'] = act_maps      

        n_hyps = frames.size(1)
        h_size = img.size(2)
        w_size = img.size(3)

        img_in = img.unsqueeze(1).repeat(1,n_hyps,1,1,1) # [b,n,c,h,w] 
        frames_in = frames.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h_size,w_size)
        focals_in = focals.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,h_size,w_size)
        # seg_map_in
        seg_mask_in = seg_mask.permute(0,3,1,2).contiguous()
        seg_mask_in = seg_mask_in.unsqueeze(1).repeat(1,n_hyps,1,1,1)
        
        act_maps_in = act_maps.permute(0,1,4,2,3).contiguous() # [b,n,c,h,w] 

        x_in = torch.cat([img_in,seg_mask_in,act_maps_in,frames_in,focals_in], dim=2)
        x_in = x_in.view(x_in.size(0)*x_in.size(1), # b*n
                         x_in.size(2), x_in.size(3), x_in.size(4))# c, h, w 
        
        logits, conv_output = self.resnet_new(x_in)
        
        logits = logits.view(-1, n_hyps, 1) # [b,n]
        out_dict['logits'] = logits
        return out_dict

class GPNet(nn.Module):
    def __init__(self, opt):
        super(GPNet, self).__init__()
        self.opt = opt
        self.isTrain = opt.isTrain
        self.gpnet_vert = GPNet_Vert(opt)

        self.frame_sampler = FrameSampler(opt)

        self.scorenet_1 = ScoreNet(opt)
        
        self.n_frames = opt.n_frames        
        self.k = opt.k
        self.use_act_score = opt.use_act_score

        self.sigmoid = nn.Sigmoid()

    def forward(self, img, segs, seg_map, seg_mask, in_dict):
        out_dict = {}

        out_dict = self.gpnet_vert(segs, out_dict=out_dict)
        zvp = out_dict['zvp']
        zvps = out_dict['zvps']

        with torch.no_grad():
            out_dict = self.frame_sampler(segs, zvp, zvps, seg_map, seg_mask, out_dict=out_dict)        
            frames = out_dict['frames']
            focals = out_dict['focals']
        
        out_dict = self.scorenet_1(img, seg_map, seg_mask, frames, focals, out_dict=out_dict)        
        scores = self.sigmoid(out_dict['logits'])        
        
        if self.isTrain == False:
            # update scores
            if self.use_act_score:
                act_maps = out_dict['act_maps']
                act_scores = gpnet_utils.compute_actmap_score(act_maps, seg_mask)
                out_dict['new_scores'] = scores = scores*act_scores
            
            if self.k > 1: 
                # average
                weights = scores
                frames, focals, weights = gpnet_utils.pick_cands_topk(frames, focals, weights, scores, k=self.k)
                out_dict['frames_inlier'] = frames
                out_dict['focals_inlier'] = focals
                out_dict['weights_inlier'] = weights
                
                weights /= weights.sum(dim=1, keepdim=True)
                                
                up_vector = gpnet_utils.average_up_vector(frames, weights)
                out_dict['up_vector'] = up_vector                  

                focal = (weights*focals).sum(dim=1)
                out_dict['focal'] = focal
            else:                
                # select best candidate
                _, best_ind = scores.max(dim=1, keepdim=True)
                  
                frame = torch.gather(frames, dim=1, index=best_ind.repeat(1,1,9))
                focal = torch.gather(focals, dim=1, index=best_ind)
                out_dict['frame'] = frame[:,0,:]
                out_dict['up_vector'] = frame[:,0,3:6]
                out_dict['focal'] = focal[:,0,:]
        
        return out_dict
