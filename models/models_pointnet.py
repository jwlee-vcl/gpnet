from __future__ import absolute_import, division, print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.k = k

        self.conv1 = torch.nn.Conv1d(self.k, 64, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(64, 128, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1, bias=False)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, k*k, bias=False)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, dtype=torch.float32, device=x.device).view(1,self.k*self.k).repeat(batchsize,1)        
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class PointNetBase(nn.Module):
    def __init__(self, ich=3, och_inst=64, och_global=1024):
        super(PointNetBase, self).__init__()
        self.och_inst   = och_inst
        self.och_global = och_global

        self.conv1 = torch.nn.Conv1d(ich, 64, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(64, self.och_inst, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(self.och_inst, 64, 1, bias=False)
        self.conv4 = torch.nn.Conv1d(64, 128, 1, bias=False)
        self.conv5 = torch.nn.Conv1d(128, self.och_global, 1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(self.och_inst)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(self.och_global)
        
        self.istn = STNkd(k=ich)
        self.fstn = STNkd(k=self.och_inst)

    def forward(self, x):        
        trans = self.istn(x)

        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        
        # [b, c, h]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        trans_feat = self.fstn(x)
        x = x.transpose(2,1)
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2,1)        

        inst_feat = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))        
        x = self.bn5(self.conv5(x))
        x = torch.max(x, 2, keepdim=True)[0]
        global_feat = x.view(-1, self.och_global)
        
        return inst_feat, global_feat 

class PointNetCls(nn.Module):
    def __init__(self, ich=3, och=2):
        super(PointNetCls, self).__init__()        
        self.feat = PointNetBase(ich=ich)
        self.fc1 = nn.Linear(1024, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, och, bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)        

    def forward(self, x):
        inst_feat, global_feat = self.feat(x)

        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNetCls2(nn.Module):
    def __init__(self, ich=1024, och=2):
        super(PointNetCls, self).__init__()                
        self.fc1 = nn.Linear(ich, 512, bias=False)
        self.fc2 = nn.Linear(512, 256, bias=False)
        self.fc3 = nn.Linear(256, och, bias=True)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)        

    def forward(self, inst_feat, global_feat):
        x = F.relu(self.bn1(self.fc1(global_feat)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x

class PointNetSeg(nn.Module):
    def __init__(self, ich=3, och=2):
        super(PointNetSeg, self).__init__()
        self.och = och
        self.feat = PointNetBase(ich=ich)
        self.conv1 = torch.nn.Conv1d(1024+64, 512, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(512, 256, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(256, 128, 1, bias=False)
        self.conv4 = torch.nn.Conv1d(128, self.och, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        
        inst_feat, global_feat = self.feat(x)

        global_feat = global_feat.view(-1, 1024, 1).repeat(1, 1, n_pts)
        x = torch.cat([inst_feat, global_feat], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)        
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_pts, self.och)
        return x

class PointNetSeg2(nn.Module):
    def __init__(self, ich_inst=64, ich_global=1024, och=2):
        super(PointNetSeg2, self).__init__()
        self.ich_inst = ich_inst
        self.ich_global = ich_global
        self.och = och
        
        self.conv1 = torch.nn.Conv1d(self.ich_inst + self.ich_global, 512, 1, bias=False)
        self.conv2 = torch.nn.Conv1d(512, 256, 1, bias=False)
        self.conv3 = torch.nn.Conv1d(256, 128, 1, bias=False)
        self.conv4 = torch.nn.Conv1d(128, self.och, 1, bias=True)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, inst_feat, global_feat):
        batchsize = inst_feat.size()[0]
        n_inst = inst_feat.size()[2]
        
        global_feat = global_feat.view(-1, self.ich_global, 1).repeat(1, 1, n_inst)
        x = torch.cat([inst_feat, global_feat], dim=1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)        
        x = x.transpose(2,1).contiguous()
        x = x.view(batchsize, n_inst, self.och)
        return x
        