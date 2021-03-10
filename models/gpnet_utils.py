from __future__ import absolute_import, division, print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_broadcast(u, v, dim=-1):
    u1, u2, u3 = torch.unbind(u, dim=dim)
    v1, v2, v3 = torch.unbind(v, dim=dim)
    return torch.stack([(u2*v3) - (u3*v2),
                        (u3*v1) - (u1*v3),
                        (u1*v2) - (u2*v1)], dim=dim)

def cosine_distance(v1, v2, dim=-1, keepdim=False):
    v1 = F.normalize(v1, p=2, dim=dim)
    v2 = F.normalize(v2, p=2, dim=dim)    
    return torch.abs((v1*v2).sum(dim=dim, keepdim=keepdim)) 

def segs2lines(segs, focal=1.0, dim=-1):
    x1,y1,x2,y2 = torch.split(segs, 1, dim=dim)
    focal = focal*torch.ones_like(x1)
    pt1 = torch.cat([x1,y1,focal], dim=dim)
    pt2 = torch.cat([x2,y2,focal], dim=dim)
    return torch.cross(pt1, pt2, dim=dim)

def normalize_lines(line, dim=-1, eps=1e-6):
    (ab,_) = torch.split(line, [2,1], dim=dim)
    denorm = torch.norm(ab, dim=dim, keepdim=True)
    denorm = torch.max(denorm, torch.tensor(eps, device=denorm.device))
    return line/denorm

def structure_tensor(params):    
    (a,b,c) = torch.unbind(params, dim=-1)
    return torch.stack([a*a, a*b,
                        b*b, b*c,
                        c*c, c*a], dim=-1)

def _outer3(x):
    N = x.shape[0]
    return torch.matmul( \
        torch.reshape(x, (N, 3, 1)), \
        torch.reshape(x, (N, 1, 3))).view((N, 9))

def structure_tensor_loss(pred, label, weight):
    '''
    pred: (N, 3)
    label: (N, 3)
    '''
    pred = _outer3(pred)
    label = _outer3(label)
    diff = (pred - label).abs()
    loss = diff.sum(dim=1) * weight
    return loss.mean()

def pick_hyps(hyps, probs, k=256, thresh=0.5):
    probs = (probs > thresh).float()
    probs = torch.where(torch.sum(probs, dim=1, keepdim=True) > 0.0, probs, probs + 1.0)

    pinds = torch.multinomial(probs, num_samples=k, replacement=True) # [b, 128]
    pinds = torch.unsqueeze(pinds, dim=-1)

    picked_hyps = torch.gather(hyps, dim=1, index=pinds.repeat(1,1,3))        
    return picked_hyps

def pick_hyps_trio(hyps1, hyps2, hyps3, probs, k=256, thresh=0.5):
    probs = (probs > thresh).float()
    probs = torch.where(torch.sum(probs, dim=1, keepdim=True) > 0.0, probs, probs + 1.0)

    pinds = torch.multinomial(probs, num_samples=k, replacement=True) # [b, 128]
    pinds = torch.unsqueeze(pinds, dim=-1)

    picked_hyps1 = torch.gather(hyps1, dim=1, index=pinds.repeat(1,1,3))    
    picked_hyps2 = torch.gather(hyps2, dim=1, index=pinds.repeat(1,1,3))    
    picked_hyps3 = torch.gather(hyps3, dim=1, index=pinds.repeat(1,1,3))        
    return (picked_hyps1, picked_hyps2, picked_hyps3)

def pick_hyps_quartet(hyps1, hyps2, hyps3, hyps4, probs, k=256, thresh=0.5):
    probs = (probs > thresh).float()
    probs = torch.where(torch.sum(probs, dim=1, keepdim=True) > 0.0, probs, probs + 1.0)

    pinds = torch.multinomial(probs, num_samples=k, replacement=True) # [b, 128]
    pinds = torch.unsqueeze(pinds, dim=-1)

    picked_hyps1 = torch.gather(hyps1, dim=1, index=pinds.repeat(1,1,3))
    picked_hyps2 = torch.gather(hyps2, dim=1, index=pinds.repeat(1,1,3))
    picked_hyps3 = torch.gather(hyps3, dim=1, index=pinds.repeat(1,1,3))    
    picked_hyps4 = torch.gather(hyps4, dim=1, index=pinds)
    return (picked_hyps1, picked_hyps2, picked_hyps3, picked_hyps4)

def pick_cands(frames, focals, weights, probs, k=128):
    pinds = torch.multinomial(probs, num_samples=k, replacement=True) # [b, 128]
    pinds = torch.unsqueeze(pinds, dim=-1)

    picked_frames = torch.gather(frames, dim=1, index=pinds.repeat(1,1,9))
    picked_focals = torch.gather(focals, dim=1, index=pinds)    
    picked_weights = torch.gather(weights, dim=1, index=pinds)
    return (picked_frames, picked_focals, picked_weights)

def pick_cands_topk(frames, focals, weights, probs, k=16):    
    _, pinds = torch.topk(probs, k=k, dim=1, sorted=False)
    # pinds = torch.unsqueeze(pinds, dim=-1)

    picked_frames = torch.gather(frames, dim=1, index=pinds.repeat(1,1,9))
    picked_focals = torch.gather(focals, dim=1, index=pinds)    
    picked_weights = torch.gather(weights, dim=1, index=pinds)
    return (picked_frames, picked_focals, picked_weights)

def generate_inter_pts(lines1, lines2, focal=1.0, n_pts=128, eps=1e-7):
    # lines [b,l,3] 
    n_lines1 = lines1.size()[1] # m
    n_lines2 = lines2.size()[1] # m

    tmp_lines1 = torch.unsqueeze(lines1, dim=-2) # [b,l,1,3]
    tmp_lines2 = torch.unsqueeze(lines2, dim=-3) # [b,1,l,3]

    pts_mat = cross_broadcast(tmp_lines1, tmp_lines2) # [b,l,l,3]
    (_,_,z_mat) = torch.unbind(pts_mat, dim=-1)
    
    const_1_0 = torch.tensor(1.0, device=z_mat.device, requires_grad=False)
    const_0_0 = torch.tensor(0.0, device=z_mat.device, requires_grad=False)

    valid_mat = torch.where(torch.abs(z_mat) < eps, const_0_0, const_1_0) # [b,l,l]    
    valid_mat = torch.where(torch.sum(valid_mat, dim=1, keepdim=True) > eps, 
                            valid_mat, valid_mat + 1.0)
    
    valid_mat = valid_mat.view(-1, n_lines1*n_lines2)
    pts_mat   = pts_mat.view(-1,   n_lines1*n_lines2, 3)
    
    inds = torch.multinomial(valid_mat, num_samples=n_pts, replacement=True) # [b, 512]
    inds = torch.unsqueeze(inds, dim=-1)

    pts  = torch.gather(pts_mat, dim=1, index=inds.expand(-1,-1,3))
    # assert not torch.isnan(pts).any()

    const_eps = torch.tensor(eps, device=pts.device, requires_grad=False)

    (_,z) = torch.split(pts, [2,1], dim=-1)
    de = torch.max(torch.abs(z), const_eps)
    z = torch.where(z < 0, -de, de)
    pts = focal*pts/z

    return pts

def init_filter_vert_segs(segs, n_lines=128):
    device = segs.device

    # collect line segs    
    lines = segs2lines(segs, focal=1.0)
    (a,b,_) = torch.unbind(lines, dim=2)
    b = torch.where(a < 0.0, b.neg(), b)
    a = torch.where(a < 0.0, a.neg(), a)    
    theta = torch.abs(torch.atan2(b,a))

    const_1_0 = torch.tensor(1.0, device=device)
    const_0_0 = torch.tensor(0.0, device=device)

    theta_v = np.radians(22.5)
    vert_cond = torch.where(theta < theta_v, const_1_0, const_0_0)
    vert_cond = torch.where(torch.sum(vert_cond, dim=1, keepdim=True) > 0.0, 
                            vert_cond, vert_cond + 1.0)
    
    inds = torch.multinomial(vert_cond, num_samples=n_lines, replacement=True) # [b, 128]
    inds = torch.unsqueeze(inds, dim=-1)
    vert_segs = torch.gather(segs,  dim=1, index=inds.expand(-1,-1,4))
    return vert_segs

def init_filter_segs_nozvp(segs, n_vert_lines=128, n_hori_lines=256):
    device = segs.device

    # angle of line segs    
    lines = segs2lines(segs, focal=1.0)
    (a,b,_) = torch.unbind(lines, dim=2)
    b = torch.where(a < 0.0, b.neg(), b)
    a = torch.where(a < 0.0, a.neg(), a)    
    theta = torch.abs(torch.atan2(b,a))

    # collect vert segs
    const_1_0 = torch.tensor(1.0, device=device)
    const_0_0 = torch.tensor(0.0, device=device)
    
    theta_v = np.radians(22.5)
    vert_cond = torch.where(theta < theta_v, const_1_0, const_0_0)
    vert_cond = torch.where(torch.sum(vert_cond, dim=1, keepdim=True) > 0.0, 
                            vert_cond, vert_cond + 1.0)
    
    inds = torch.multinomial(vert_cond, num_samples=n_vert_lines, replacement=True) # [b, 128]
    inds = torch.unsqueeze(inds, dim=-1)
    vert_segs = torch.gather(segs,  dim=1, index=inds.expand(-1,-1,4))

    # collect hori segs
    hori_cond = torch.where(theta > theta_v, const_1_0, const_0_0)
    hori_cond = torch.where(torch.sum(hori_cond, dim=1, keepdim=True) > 0.0, 
                            hori_cond, hori_cond + 1.0)

    inds = torch.multinomial(hori_cond, num_samples=n_hori_lines, replacement=True) # [b, 128]
    inds = torch.unsqueeze(inds, dim=-1)
    hori_segs = torch.gather(segs,  dim=1, index=inds.expand(-1,-1,4))

    return vert_segs, hori_segs

def init_filter_segs(segs, vert_vp, n_vert_lines=128, n_hori_lines=256, focal=1.0, eps=1e-6):
    '''
    segs [b, n, 4]
    vert_vps [b, 3]
    '''
    device = segs.device

    # update focal
    (xy,z) = torch.split(vert_vp, [2,1], dim=-1) 
    z = focal*torch.ones_like(z)
    vert_vp = torch.cat([xy,z], dim=-1)

    lines = segs2lines(segs, focal=focal) # [b,n,3]
    tmp_vert_vp = torch.unsqueeze(vert_vp, dim=-2) # [b,1,3]

    # lines near the vertical vps (vert line segs)
    dist = cosine_distance(lines, tmp_vert_vp) # [b,n]
    
    thresh = np.radians(2.0)
    const_1_0 = torch.tensor(1.0, device=device)
    const_0_0 = torch.tensor(0.0, device=device)

    vert_cond = torch.where(dist < thresh, const_1_0, const_0_0)
    vert_cond = torch.where(torch.sum(vert_cond, dim=1, keepdim=True) > eps, 
                            vert_cond, vert_cond + 1.0)
    inds = torch.multinomial(vert_cond, num_samples=n_vert_lines, replacement=True) # [b, 128]
    inds = torch.unsqueeze(inds, dim=-1)
    vert_segs = torch.gather(segs, dim=1, index=inds.repeat(1,1,4))
    
    # other lines (hori line segs)
    hori_cond = torch.where(dist < thresh, const_0_0, const_1_0)
    hori_cond = torch.where(torch.sum(hori_cond, dim=1, keepdim=True) > eps, 
                            hori_cond, hori_cond + 1.0)
    inds = torch.multinomial(hori_cond, num_samples=n_hori_lines, replacement=True) # [b, 128]
    inds = torch.unsqueeze(inds, dim=-1)
    hori_segs = torch.gather(segs, dim=1, index=inds.repeat(1,1,4))

    return vert_segs, hori_segs

def junc_filtering(vert_segs, hori_segs, n_juncs=512, focal=1.0, eps=1e-6):    
    device = vert_segs.device
    
    num_vert_segs = vert_segs.size()[1] # n
    num_hori_segs = hori_segs.size()[1] # m
    
    vert_lines = segs2lines(vert_segs, focal=focal)
    hori_lines = segs2lines(hori_segs, focal=focal)
    
    vert_lines = normalize_lines(vert_lines)
    hori_lines = normalize_lines(hori_lines)
        
    tmp_vert_segs = torch.unsqueeze(vert_segs, dim=-2) # [b,n,1,4]
    tmp_hori_segs = torch.unsqueeze(hori_segs, dim=-3) # [b,1,m,4]
    
    tmp_vert_lines = torch.unsqueeze(vert_lines, dim=-2) # [b,n,1,3]
    tmp_hori_lines = torch.unsqueeze(hori_lines, dim=-3) # [b,1,m,4]
    
    pts_mat = cross_broadcast(tmp_vert_lines, tmp_hori_lines)
    (_, z_mat) = torch.split(pts_mat, [2,1], dim=-1)
    pts_mat = torch.where(z_mat < 0.0, pts_mat.neg(), pts_mat)
    pts_mat /= torch.max(torch.abs(z_mat), torch.tensor(eps, device=device))
    
    (xy_mat,_) = torch.split(pts_mat, [2,1], dim=-1)
    
    (vert_pt1, vert_pt2) = torch.split(tmp_vert_segs, [2,2], dim=-1) # [b,n,1,2]
    (hori_pt1, hori_pt2) = torch.split(tmp_hori_segs, [2,2], dim=-1) # [b,1,m,2]
    
    dist_mat_vert_1 = torch.norm(xy_mat - vert_pt1, dim=-1) # [b,n,m]
    dist_mat_vert_2 = torch.norm(xy_mat - vert_pt2, dim=-1) # [b,n,m]
    dist_mat_hori_1 = torch.norm(xy_mat - hori_pt1, dim=-1) # [b,n,m]
    dist_mat_hori_2 = torch.norm(xy_mat - hori_pt2, dim=-1) # [b,n,m]
        
    const_1_0 = torch.tensor(1.0, device=device, requires_grad=False)
    const_0_0 = torch.tensor(0.0, device=device, requires_grad=False)

    thresh = 4.0/320.0
    valid_mat = torch.where(
        (torch.lt(dist_mat_vert_1, thresh) | torch.lt(dist_mat_vert_2, thresh)) |
        (torch.lt(dist_mat_hori_1, thresh) | torch.lt(dist_mat_hori_2, thresh)), 
        const_1_0, const_0_0) # [b,n,m]
    
    vert_lines_mat = tmp_vert_lines.repeat(1,1,num_hori_segs,1) # [n,m,3]
    hori_lines_mat = tmp_hori_lines.repeat(1,num_vert_segs,1,1) # [n,m,3]
        
    vert_segs_mat = tmp_vert_segs.repeat(1,1,num_hori_segs,1) # [n,m,4]
    hori_segs_mat = tmp_hori_segs.repeat(1,num_vert_segs,1,1) # [n,m,4]

    # select
    valid_mat = valid_mat.view(-1, num_vert_segs*num_hori_segs) # [b, n*m]
    # prevent zero-sum    
    valid_mat = torch.where(torch.sum(valid_mat, dim=1, keepdim=True) > eps, 
                            valid_mat, valid_mat + 1.0)
    
    pts_mat = pts_mat.view(-1, num_vert_segs*num_hori_segs, 3) # [b, n*m, 3]
    
    vert_segs_mat = vert_segs_mat.view(-1, num_vert_segs*num_hori_segs, 4) # [b, n*m, 4]
    hori_segs_mat = hori_segs_mat.view(-1, num_vert_segs*num_hori_segs, 4) # [b, n*m, 4]

    vert_lines_mat = vert_lines_mat.view(-1, num_vert_segs*num_hori_segs, 3) # [b, n*m, 3]
    hori_lines_mat = hori_lines_mat.view(-1, num_vert_segs*num_hori_segs, 3) # [b, n*m, 3]
    
    inds = torch.multinomial(valid_mat, num_samples=n_juncs, replacement=True) # [b, 512]
    inds = torch.unsqueeze(inds, dim=-1)
    
    pts  = torch.gather(pts_mat, dim=1, index=inds.expand(-1,-1,3))
    vert_segs  = torch.gather(vert_segs_mat,  dim=1, index=inds.expand(-1,-1,4))
    vert_lines = torch.gather(vert_lines_mat, dim=1, index=inds.expand(-1,-1,3))
    hori_segs  = torch.gather(hori_segs_mat,  dim=1, index=inds.expand(-1,-1,4))
    hori_lines = torch.gather(hori_lines_mat, dim=1, index=inds.expand(-1,-1,3))
    
    return (pts, vert_segs, vert_lines, hori_segs, hori_lines)

def filter_lines_by_dir(hori_segs, hori_lines, hl, n_lines=256, eps=1e-7):
    '''
    hori_lines: [b,n,3]
    hl: [b,3]
    '''
    (a,b,c) = torch.split(hl, 1, dim=1)
    hl = torch.where(c < 0.0, hl.neg(), hl)
    
    # tmp_hl = tf.expand_dims(hl, axis=-2) # [1,3]    
    tmp_hl = hl.view(-1,1,3) # [1,3]    
    
    # line - hl inter points
    hori_pts = cross_broadcast(hori_lines, tmp_hl) # [b,n,3] 
    
    (x,y,z) = torch.split(hori_pts, 1, dim=2)
    hori_pts = torch.where(z < 0.0, hori_pts.neg(), hori_pts)
    
    (x,y,_) = torch.unbind(hori_pts, dim=2)    
    (a,b,_) = torch.unbind(tmp_hl, dim=2)
    dirs = b*x - a*y # [b,n] 
    
    const_1_0 = torch.tensor(1.0, device=dirs.device, requires_grad=False)
    const_0_0 = torch.tensor(0.0, device=dirs.device, requires_grad=False)

    pos_mat = torch.where(torch.gt(dirs, 0.0), const_1_0, const_0_0)
    pos_mat = torch.where(torch.sum(pos_mat, dim=1, keepdim=True) > eps, 
                          pos_mat, pos_mat + 1.0)

    pos_inds = torch.multinomial(pos_mat, num_samples=n_lines, replacement=True) # [b, 512]
    pos_inds = torch.unsqueeze(pos_inds, dim=-1)

    hori_pos_segs  = torch.gather(hori_segs, dim=1, index=pos_inds.expand(-1,-1,4))

    neg_mat = torch.where(torch.le(dirs, 0), const_1_0, const_0_0)
    neg_mat = torch.where(torch.sum(neg_mat, dim=1, keepdim=True) > eps, 
                          neg_mat, neg_mat + 1.0)

    neg_inds = torch.multinomial(neg_mat, num_samples=n_lines, replacement=True) # [b, 512]
    neg_inds = torch.unsqueeze(neg_inds, dim=-1)

    hori_neg_segs  = torch.gather(hori_segs, dim=1, index=neg_inds.expand(-1,-1,4))
        
    return (hori_pos_segs, hori_neg_segs)

def generage_frames(zvp, hpts_p, hpts_m, n_frames=512, eps=1e-7):
    '''
    zvp : [b,3]
    hpts_p : [b,n,3]
    hpts_m : [b,n,3]
    '''
    thresh_pitch = np.radians(30.0, dtype=np.float32)

    n_hpts_p = hpts_p.size(1)
    n_hpts_m = hpts_m.size(1)

    zvp = proj_pts(zvp)
    hpts_p = proj_pts(hpts_p)
    hpts_m = proj_pts(hpts_m)

def proj_pts(pts, eps=1e-7):
    device = pts.device
    (x,y,z) = torch.split(pts, 1, dim=-1)
    de = torch.max(torch.abs(z), torch.tensor(eps, device=device))
    de = torch.where(z < 0.0, de.neg(), de)
    return pts/de

def compute_focal(zvps, hpts, f_min=1.0, f_max=3.0):
    '''
    zvps : [b,m,3]
    hpts : [b,n,3]
    ''' 
    device = zvps.device

    f_min_sq = f_min**2
    f_max_sq = f_max**2

    zvps = torch.unsqueeze(zvps, dim=1) # [b,1,m,3]
    hpts = torch.unsqueeze(hpts, dim=2) # [b,n,1,3]

    a,b,_ = torch.unbind(zvps, dim=3)
    x,y,_ = torch.unbind(hpts, dim=3)
    f2_mat = (a*x + b*y).neg() # [b,n,m]
    f_mat = F.relu(f2_mat).sqrt()
    
    const_0_0 = torch.tensor(0.0, device=device)
    const_1_0 = torch.tensor(1.0, device=device)    
    valid_mat = torch.where(torch.lt(f2_mat, f_min_sq) | torch.gt(f2_mat, f_max_sq),
                            const_0_0, const_1_0)    
    return f_mat, valid_mat # [b,n,m]

def update_focal(v, f, dim=-1):
    (x,y,_) = torch.unbind(v, dim=dim)
    return torch.stack([x,y,f], dim=dim)    

def flip_dirs(dirs, dim=2):
    _,_,z = torch.split(dirs, 1, dim=dim)
    return torch.where(z < 0, dirs.neg(), dirs)
         
def generate_frames(zvps, hpts_p, hpts_m, n_frames=256):
    '''
    zvps : [b,m,3]
    hpts_p : [b,n,3]
    hpts_m : [b,n,3]
    '''
    n_zvps = zvps.size(1)
    n_hpts = hpts_p.size(1)
    
    zvps = proj_pts(zvps)
    hpts_p = proj_pts(hpts_p)
    hpts_m = proj_pts(hpts_m)

    focals_p, vmat_p = compute_focal(zvps, hpts_p) # [b,n,m]
    focals_m, vmat_m = compute_focal(zvps, hpts_m) # [b,n,m]
    
    focals_p = focals_p.view(focals_p.size(0), focals_p.size(1)*focals_p.size(2))    
    focals_m = focals_m.view(focals_m.size(0), focals_m.size(1)*focals_m.size(2))
    vmat_p = vmat_p.view(vmat_p.size(0), vmat_p.size(1)*vmat_p.size(2))    
    vmat_m = vmat_m.view(vmat_m.size(0), vmat_m.size(1)*vmat_m.size(2))
    
    zvps = torch.unsqueeze(zvps, dim=1) # [b,1,m,3]
    zvps = zvps.repeat(1,n_hpts,1,1) # [b,n,m,3]
    zvps = zvps.view(zvps.size(0), zvps.size(1)*zvps.size(2), 3)

    hpts_p = torch.unsqueeze(hpts_p, dim=2) # [b,n,1,3]
    hpts_p = hpts_p.repeat(1,1,n_zvps,1) # [b,n,m,3]
    hpts_p = hpts_p.view(hpts_p.size(0), hpts_p.size(1)*hpts_p.size(2), 3)

    hpts_m = torch.unsqueeze(hpts_m, dim=2) # [b,n,1,3]
    hpts_m = hpts_m.repeat(1,1,n_zvps,1) # [b,n,m,3]
    hpts_m = hpts_m.view(hpts_m.size(0), hpts_m.size(1)*hpts_m.size(2), 3)

    zvps_p = update_focal(zvps, focals_p) # [b,n,3]
    zvps_m = update_focal(zvps, focals_m)
        
    hpts_p = update_focal(hpts_p, focals_p)
    hpts_m = update_focal(hpts_m, focals_m)

    ydirs_p = F.normalize(zvps_p, dim=2)
    ydirs_m = F.normalize(zvps_m, dim=2)

    xdirs_p = F.normalize(hpts_p, dim=2)
    zdirs_m = F.normalize(hpts_m, dim=2)

    zdirs_p = torch.cross(ydirs_p, xdirs_p, dim=2)
    xdirs_m = torch.cross(zdirs_m, ydirs_m, dim=2)

    zdirs_p = flip_dirs(zdirs_p)
    xdirs_m = flip_dirs(xdirs_m)
    
    xdirs = torch.cat([xdirs_p, xdirs_m], dim=1)
    ydirs = torch.cat([ydirs_p, ydirs_m], dim=1)
    zdirs = torch.cat([zdirs_p, zdirs_m], dim=1)
    focals = torch.cat([focals_p, focals_m], dim=1)
    focals = focals.unsqueeze(-1)
    vmat = torch.cat([vmat_p, vmat_m], dim=1)
    
    # xdirs, ydirs, zdirs = pick_hyps_trio(xdirs, ydirs, zdirs, vmat, k=n_frames)
    xdirs, ydirs, zdirs, focals = pick_hyps_quartet(xdirs, ydirs, zdirs, focals, vmat, k=n_frames)
    
    frames = torch.cat([xdirs, ydirs, zdirs], dim=2)
    return frames, focals

def segs2lines_map(segs, focals=1.0):
    x1,y1,x2,y2 = torch.split(segs, 1, dim=-1)    
    pt1 = torch.cat([x1,y1,focals], dim=-1)
    pt2 = torch.cat([x2,y2,focals], dim=-1)
    return torch.cross(pt1, pt2, dim=-1)

def generate_active_map(frames, focals, seg_map, seg_mask):
    '''
    frames [b,n,9]
    focals [b,n,1]
    seg_map [b,h,w,4]
    seg_mask [b,h,w,1]

    out [b,n,h,w,1] + [b,n,h,w,9]
    '''
    n_cands = frames.size(1)
    map_h = seg_map.size(1)
    map_w = seg_map.size(2)

    sigma_l2p = np.cos(np.radians(90.0 - 1.0), dtype=np.float32) # near 0.0        
    sigma_l2p = 1.0/(2*np.square(sigma_l2p))

    # prepare vps
    frames = frames.view(-1, n_cands, 1, 1, 9)    
    frames_mat = frames.repeat(1, 1, map_h, map_w, 1) # [b,n,h,w,9]    
    (vpx, vpy, vpz) = torch.split(frames_mat, 3, dim=-1) # [b,n,h,w,3] x 3

    # prepare line maps 
    focals = focals.view(-1, n_cands, 1, 1, 1)
    focals = focals.repeat(1, 1, map_h, map_w, 1) # [b,n,h,w,1]    
    
    seg_map = seg_map.unsqueeze(1) # [b,1,h,w,4]
    seg_map = seg_map.repeat(1, n_cands, 1, 1, 1) # [b,n,h,w,4]
    line_map = segs2lines_map(seg_map, focals) # [b,n,h,w,3]

    seg_mask = seg_mask.unsqueeze(1) # [b,1,h,w,1]
    seg_mask = seg_mask.repeat(1, n_cands, 1, 1, 1) # [b,n,h,w,1]

    active_map_x = cosine_distance(line_map, vpx, keepdim=True) # [good:0 ~ 1:bad]
    active_map_x = seg_mask*(1.0 - active_map_x) # [b,n,h,w,1]

    active_map_y = cosine_distance(line_map, vpy, keepdim=True)
    active_map_y = seg_mask*(1.0 - active_map_y)

    active_map_z = cosine_distance(line_map, vpz, keepdim=True)
    active_map_z = seg_mask*(1.0 - active_map_z)

    active_map = torch.cat([active_map_x, active_map_y, active_map_z], dim=-1) # [b,n,h,w,3]
    # active_map, _ = torch.max(active_map, dim=-1, keepdim=True) # [b,n,h,w,1]
    active_map = torch.exp(-(1.0 - active_map).pow(2.0)*sigma_l2p) # [b,n,h,w,1]
    
    # cond_map = torch.cat([active_map, frames_mat]) # [b,n,h,w,10]
    return active_map, frames_mat, seg_mask 

def compute_actmap_score(actmap, seg_mask):
    actmap, _ = actmap.max(dim=-1, keepdim=True) # [b,n,h,w,1]
    seg_mask = seg_mask.unsqueeze(dim=1)    
    numer = actmap.sum(dim=2).sum(dim=2)
    denorm = seg_mask.sum(dim=2).sum(dim=2)    
    scores = numer/denorm # (bad) 0 ~ 1 (good) # [b,n,1]
    return scores 

def solving_pose(frames, weights):
    frames = frames.transpose(2,1).contiguous()
    weights = weights.transpose(2,1).contiguous()

    num_samples = frames.size(0)
    num_frames = frames.size(2)

    up_vector = torch.from_numpy(np.array([0.0, 1.0, 0.0], dtype=np.float32)).cuda()
    up_vector = torch.reshape(up_vector, [1,3,1]).repeat(num_samples, 1, num_frames)

    identity_mat = torch.eye(3).float().cuda()
    identity_mat_rep = identity_mat.unsqueeze(0).repeat(num_samples,1,1)

    weights_x = weights[:, 0:1, :].repeat(1,3,1)
    weights_y = weights[:, 1:2, :].repeat(1,3,1)
    weights_z = weights[:, 2:3, :].repeat(1,3,1)

    frames_x = frames[:, 0:3, :] 
    frames_y = frames[:, 3:6, :]
    frames_z = frames[:, 6:9, :]

    frames_x_w = frames_x * weights_x
    frames_y_w = frames_y * weights_y
    frames_z_w = frames_z * weights_z

    # M * 3 x 3N matrix
    A_w = torch.cat((frames_x_w, frames_y_w, frames_z_w), dim=2)

    up_vector_w = weights*up_vector
    # M * 1 * 3N
    b_w = torch.cat((up_vector_w[:, 0:1, :], up_vector_w[:, 1:2, :], up_vector_w[:, 2:3, :]), dim=2)

    # M*3*3
    H = torch.bmm(A_w, torch.transpose(A_w, 1, 2))
    # M*3*1
    g = torch.bmm(A_w, torch.transpose(b_w, 1, 2))
    ggT = torch.bmm(g, torch.transpose(g, 1, 2))

    # A0 = torch.bmm(H, H) - ggT
    # A1 = -2.0 * H

    C_mat = torch.cat( (torch.cat((H, -identity_mat_rep), dim=2), torch.cat((-ggT, H), dim=2)), dim=1)
    
    est_up_list = []

    for i in range(num_samples):
        est_lambda = torch.eig(C_mat[i, :, :])
        est_lambda = est_lambda[0]

        img_part = est_lambda[:, 1]
        real_part = est_lambda[:, 0]

        min_lambda = torch.min(real_part[torch.abs(img_part.data) < 1e-6]).item()

        est_up_n = torch.matmul(torch.pinverse(H[i, :, :] - min_lambda * identity_mat), g[i, :, :])
        est_up_n = est_up_n[:, 0]

        est_up_list.append(est_up_n)

    return torch.stack(est_up_list, dim=0)

def average_up_vector(frames, weights):
    weights /= weights.sum(dim=1, keepdim=True)

    up = frames[:, :, 3:6]
    b = up[:,:,1:2]

    up = torch.where(b < 0, up.neg(), up)

    # average dir (a,b)
    up_vector = torch.sum(weights*up, dim=1)
    up_vector = F.normalize(up_vector, dim=1)
    return up_vector
        


