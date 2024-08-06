import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mutual_info_score
from torch.nn import functional as F
import gl
from metric_select import metric_select
from backbone_select import Backbone
from mlpmixer import MLP_Mix_Enrich
from mlpmixer import MLP_Mix_Joint


def MI(x, y, bins):
    # Estimate the mutual information between x and y using histograms
    c_xy = np.histogram2d(x, y, bins)[0]
    mi = mutual_info_score(None, None, contingency=c_xy)
    return mi


def loss_mi_optimized(x, n_class, n_sup):
    loss = torch.tensor(0).float().to(gl.device)
    x = x.clone().detach()
    x = x.view(n_class * n_sup, -1).cpu().numpy()
    bins = 10
    MI_matrix = np.zeros(n_class * n_sup)
    for i in range(n_class * n_sup):
        MI_matrix[i] = MI(x[i, :], x[i, :], bins)
    MI_matrix = torch.from_numpy(MI_matrix)
    for k in range(n_class):
        numerator = torch.sum((MI_matrix[(k * n_sup):((k + 1) * n_sup)])) / 2
        loss += torch.log(numerator / 1e-6)
    return loss / n_class


class ProtoNet(nn.Module):

    def __init__(self, opt):
        super(ProtoNet, self).__init__()

        if gl.dataset == 'kinetics_2d':
            ch = 2
        else:
            ch = 3

        self.model, self.out_channel, self.seq_len, node = Backbone(gl.dataset, gl.backbone, channel=ch)

        self.trans_linear_in_dim = node * self.out_channel

        if gl.mix == 1:
            self.fr_enrich = MLP_Mix_Enrich(self.trans_linear_in_dim, self.seq_len, gl.pe1)
        else:
            self.fr_enrich = None

        if gl.mixjoint == 1:
            self.joint_enrich = MLP_Mix_Joint(self.out_channel, node, gl.pe2)
        else:
            self.joint_enrich = None

    def train_mode(self, input, target, n_support):
        # input is encoder by ST_GCN
        n, c, t, v = input.size()

        def supp_idxs(cc):
            return torch.where(target.eq(cc))[0][:n_support]

        # FIXME when torch.unique will be available on cuda too
        classes = torch.unique(target)
        n_class = len(classes)
        n_query = torch.where(target.eq(classes[0]))[0].size(0) - n_support

        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)
        if gl.mi != 0:
            miloss = loss_mi_optimized(z_proto, n_class, n_support)
        else:
            miloss = torch.tensor(0).float().to(gl.device)

        query_idxs = torch.stack(list(map(lambda c: torch.where(target.eq(c))[0][n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]  # n是样本数, c, t, v
        # z_proto.shape, zq.shape:torch.Size([5, 256, 8, 25]) torch.Size([50, 256, 8, 25]) n,c,t,v
        if gl.sat == 1:
            if gl.mixjoint == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query * t, v, c)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support * t, v, c)
                zq = self.joint_enrich(zq)
                z_proto = self.joint_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
            if gl.mix == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query, t, -1)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support, t, -1)
                zq = self.fr_enrich(zq)
                z_proto = self.fr_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
        elif gl.sat == 2:
            if gl.mix == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query, t, -1)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support, t, -1)
                zq = self.fr_enrich(zq)
                z_proto = self.fr_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
            if gl.mixjoint == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query * t, v, c)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support * t, v, c)
                zq = self.joint_enrich(zq)
                z_proto = self.joint_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
        # 形成类原型
        z_proto = z_proto.reshape(n_class, n_support, c, t, v).mean(1)  # n是类数, c, t, v

        dist, z_proto, zq = metric_select(z_proto, zq, gl.metric, n_class, n_query, t, v, c, gl.m1, gl.m2)

        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)
        target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()
        # log_p_y.shape: ([5, 10, 5]),target_inds.shape: ([5, 10, 1])
        _, y_hat = log_p_y.max(2)
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        if gl.mi != 0:
            loss_val = loss_val + miloss * gl.mi
        return loss_val, acc_val, miloss

    def evaluate(self, input, target, n_support):
        n, c, t, v = input.size()
        classes = torch.unique(target)
        n_class = len(classes)
        n_query = torch.where(target.eq(classes[0]))[0].size(0) - n_support

        def supp_idxs(cc):
            return torch.where(target.eq(cc))[0][:n_support]

        support_idxs = list(map(supp_idxs, classes))
        z_proto = torch.stack([input[idx_list] for idx_list in support_idxs]).view(-1, c, t, v)

        query_idxs = torch.stack(list(map(lambda c: torch.where(target.eq(c))[0][n_support:], classes))).view(-1)
        zq = input[query_idxs.long()]

        if gl.sat == 1:
            if gl.mixjoint == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query * t, v, c)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support * t, v, c)
                zq = self.joint_enrich(zq)
                z_proto = self.joint_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
            if gl.mix == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query, t, -1)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support, t, -1)
                zq = self.fr_enrich(zq)
                z_proto = self.fr_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
        elif gl.sat == 2:
            if gl.mix == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query, t, -1)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support, t, -1)
                zq = self.fr_enrich(zq)
                z_proto = self.fr_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
            if gl.mixjoint == 1:
                zq = zq.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_query * t, v, c)
                z_proto = z_proto.permute(0, 2, 3, 1).contiguous().reshape(n_class * n_support * t, v, c)
                zq = self.joint_enrich(zq)
                z_proto = self.joint_enrich(z_proto)
                zq = zq.reshape(n_class * n_query, t, v, c).permute(0, 3, 1, 2).contiguous()
                z_proto = z_proto.reshape(n_class * n_support, t, v, c).permute(0, 3, 1, 2).contiguous()
        # 形成类原型
        z_proto = z_proto.reshape(n_class, n_support, c, t, v).mean(1)  # n是类数, c, t, v

        dist, z_proto, zq = metric_select(z_proto, zq, gl.metric, n_class, n_query, t, v, c, gl.m1, gl.m2)

        log_p_y = F.log_softmax(-dist, dim=1).view(n_class, n_query, -1)
        target_inds = torch.arange(0, n_class).to(gl.device)
        target_inds = target_inds.view(n_class, 1, 1)
        target_inds = target_inds.expand(n_class, n_query, 1).long()

        _, y_hat = log_p_y.max(2)
        acc_val = y_hat.eq(target_inds.squeeze()).float().mean()

        return acc_val

    def forward(self, x):
        x = self.model(x)
        return x
