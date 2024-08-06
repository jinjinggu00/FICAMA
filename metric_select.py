import torch
from torch.nn import functional as F
from metric_function.metric_function import euclidean_dist, chebyshev_dist, jaccard_dist, manhattan_dist, cosine_dist, \
    dtw_loss, \
    bimmh_dist, otam, tsm_painet
from utils import dist_normalization


def metric_select(z_proto, zq, metric, n_class, n_query, t, v, c, m1='tcmhm', m2='eucl'):
    # 选择距离度量
    if metric == 'dtw':
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        zq = zq.permute(0, 2, 3, 1).contiguous()
        dist = dtw_loss(zq, z_proto)
    elif metric == 'eucl':
        zq = zq.reshape(n_class * n_query, -1)
        z_proto = z_proto.view(n_class, -1)
        dist = euclidean_dist(zq, z_proto)
    elif metric == 'cosine':
        zq = zq.reshape(n_class * n_query, -1)
        z_proto = z_proto.view(n_class, -1)
        dist = cosine_dist(zq, z_proto)
    elif metric == 'manhattan':
        zq = zq.reshape(n_class * n_query, -1)
        z_proto = z_proto.view(n_class, -1)
        dist = manhattan_dist(zq, z_proto)
    elif metric == 'chebyshev':
        "切比雪夫距离"
        zq = zq.reshape(n_class * n_query, -1)
        z_proto = z_proto.view(n_class, -1)
        dist = chebyshev_dist(zq, z_proto)
    elif metric == 'jaccard':
        "杰卡德距离"
        zq = zq.reshape(n_class * n_query, -1)
        z_proto = z_proto.view(n_class, -1)
        dist = jaccard_dist(zq, z_proto)
    elif metric == 'tcmhm':
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        # print(zq.shape, z_proto.shape)#orch.Size([50, 256, 8, 25]) torch.Size([5, 256, 8, 25])
        dist = bimmh_dist(zq, z_proto)  # torch.Size([50, 5]
        # 将dist中的nan置为0
        if torch.isnan(dist).any():
            print('dist存在nan')
            # 输出nan的个数
            print(torch.count_nonzero(torch.isnan(dist)))
            dist = torch.nan_to_num(dist, nan=1e-6)
        dist = dist_normalization(dist)
    elif metric == 'mhm':
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        # print(zq.shape, z_proto.shape)#orch.Size([50, 256, 8, 25]) torch.Size([5, 256, 8, 25])
        dist = bimmh_dist(zq, z_proto, 0)  # torch.Size([50, 5]
        if torch.isnan(dist).any():
            print('dist存在nan')
            print(torch.count_nonzero(torch.isnan(dist)))
            dist = torch.nan_to_num(dist, nan=1e-6)
        dist = dist_normalization(dist)
    elif metric == 'tsm':
        zq = zq.permute(0, 2, 3, 1).contiguous()  # n, t, v, c
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        dist = tsm_painet(zq, z_proto)  # torch.Size([50, 5]
        if torch.isnan(dist).any():
            print('dist存在nan')
            print(torch.count_nonzero(torch.isnan(dist)))
            dist = torch.nan_to_num(dist, nan=1e-6)
        dist = dist_normalization(dist)
    elif metric == 'otam':
        zq = zq.permute(0, 2, 3, 1).contiguous()
        z_proto = z_proto.permute(0, 2, 3, 1).contiguous()
        if len(zq.size()) == 2:
            zq = zq.view(n_class * n_query, t, v, c)
            z_proto = z_proto.view(n_class, t, v, c)
        dist = otam(zq, z_proto)
    elif metric == 'en1':
        # 基于概率矩阵中的每一行的最大值和次大值的差值作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)
        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)
        # Softmax probability for each metric
        softmax_prob1 = F.softmax(-dist1, dim=1)  # shape:torch.Size([50, 5])
        softmax_prob2 = F.softmax(-dist2, dim=1)

        # Entropy of probability distribution for each metric
        entropy1 = -(softmax_prob1 * torch.log(softmax_prob1)).sum(dim=1)  # shape:torch.Size([50])
        entropy2 = -(softmax_prob2 * torch.log(softmax_prob2)).sum(dim=1)

        # Margin between top predictions for each metric
        _, top2_indices1 = torch.topk(-dist1, k=2, dim=1)
        top2_probs1 = softmax_prob1.gather(1, top2_indices1)
        margin1 = top2_probs1[:, 0] - top2_probs1[:, 1]

        _, top2_indices2 = torch.topk(-dist2, k=2, dim=1)
        top2_probs2 = softmax_prob2.gather(1, top2_indices2)
        margin2 = top2_probs2[:, 0] - top2_probs2[:, 1]

        # Calculate weights based on margin (you can use other metrics like entropy)
        # margin1.shape:torch.Size([50]), margin2.shape:torch.Size([50])
        weights = torch.stack([margin1, margin2], dim=1)
        # weights.shape:torch.Size([50, 2])
        weights = F.softmax(weights, dim=1)
        # weights.shape:torch.Size([50, 2])

        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en2':
        # 基于距离矩阵中的每一行的最大值和次大值的差值作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)

        # Margin between top predictions for each metric
        _, top2_indices1 = torch.topk(-dist1, k=2, dim=1)
        top2_probs1 = dist1.gather(1, top2_indices1)
        margin1 = top2_probs1[:, 1] - top2_probs1[:, 0]

        _, top2_indices2 = torch.topk(-dist2, k=2, dim=1)
        top2_probs2 = dist2.gather(1, top2_indices2)
        margin2 = top2_probs2[:, 1] - top2_probs2[:, 0]

        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([margin1, margin2], dim=1)
        weights = F.softmax(weights, dim=1)

        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en3':
        # 基于负自熵作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)
        # Softmax probability for each metric
        softmax_prob1 = F.softmax(-dist1, dim=1)  # shape:torch.Size([50, 5])
        softmax_prob2 = F.softmax(-dist2, dim=1)

        # Entropy of probability distribution for each metric
        entropy1 = -(softmax_prob1 * torch.log(softmax_prob1)).sum(dim=1)  # shape:torch.Size([50])
        entropy2 = -(softmax_prob2 * torch.log(softmax_prob2)).sum(dim=1)
        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([entropy1, entropy2], dim=1)
        weights = F.softmax(weights, dim=1)
        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en4':
        # 基于后验分布的每一行的最大值作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)
        # Softmax probability for each metric
        softmax_prob1 = F.softmax(-dist1, dim=1)  # shape:torch.Size([50, 5])
        softmax_prob2 = F.softmax(-dist2, dim=1)
        # 找出softmax_prob中每行的最大值，获得一个torch.Size([50])的张量
        max1 = torch.max(softmax_prob1, dim=1).values
        max2 = torch.max(softmax_prob2, dim=1).values
        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([max1, max2], dim=1)
        weights = F.softmax(weights, dim=1)
        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en5':
        # 基于概率矩阵中的每一行的最大值和次大值的差值作为权重,使用平均值而不是softmax
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)
        # Softmax probability for each metric
        softmax_prob1 = F.softmax(-dist1, dim=1)  # shape:torch.Size([50, 5])
        softmax_prob2 = F.softmax(-dist2, dim=1)

        # Margin between top predictions for each metric
        _, top2_indices1 = torch.topk(-dist1, k=2, dim=1)
        top2_probs1 = softmax_prob1.gather(1, top2_indices1)
        margin1 = top2_probs1[:, 0] - top2_probs1[:, 1]

        _, top2_indices2 = torch.topk(-dist2, k=2, dim=1)
        top2_probs2 = softmax_prob2.gather(1, top2_indices2)
        margin2 = top2_probs2[:, 0] - top2_probs2[:, 1]

        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([margin1, margin2], dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en6':
        # 基于距离矩阵中的每一行的最大值和次大值的差值作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)

        # Margin between top predictions for each metric
        _, top2_indices1 = torch.topk(-dist1, k=2, dim=1)
        top2_probs1 = dist1.gather(1, top2_indices1)
        margin1 = top2_probs1[:, 1] - top2_probs1[:, 0]

        _, top2_indices2 = torch.topk(-dist2, k=2, dim=1)
        top2_probs2 = dist2.gather(1, top2_indices2)
        margin2 = top2_probs2[:, 1] - top2_probs2[:, 0]

        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([margin1, margin2], dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)

        # weights = torch.full_like(weights, 0.5)        平均加权的方法
        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en7':
        # 基于负自熵作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)
        # Softmax probability for each metric
        softmax_prob1 = F.softmax(-dist1, dim=1)  # shape:torch.Size([50, 5])
        softmax_prob2 = F.softmax(-dist2, dim=1)

        # Entropy of probability distribution for each metric
        entropy1 = -(softmax_prob1 * torch.log(softmax_prob1)).sum(dim=1)  # shape:torch.Size([50])
        entropy2 = -(softmax_prob2 * torch.log(softmax_prob2)).sum(dim=1)
        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([entropy1, entropy2], dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    elif metric == 'en8':
        # 基于后验分布的每一行的最大值作为权重
        valid_values = {'en1', 'en2', 'en3', 'en4', 'en5', 'en6', 'en7', 'en8'}
        if m1 in valid_values or m2 in valid_values:
            raise ValueError('You cannot call integration metrics again in integration metrics as this may cause '
                             'stack overflow')
        dist1, _, _ = metric_select(z_proto, zq, m1, n_class, n_query, t, v, c)
        dist2, _, _ = metric_select(z_proto, zq, m2, n_class, n_query, t, v, c)

        dist1 = dist_normalization(dist1)
        dist2 = dist_normalization(dist2)
        # Softmax probability for each metric
        softmax_prob1 = F.softmax(-dist1, dim=1)  # shape:torch.Size([50, 5])
        softmax_prob2 = F.softmax(-dist2, dim=1)
        # 找出softmax_prob中每行的最大值，获得一个torch.Size([50])的张量
        max1 = torch.max(softmax_prob1, dim=1).values
        max2 = torch.max(softmax_prob2, dim=1).values
        # Calculate weights based on margin (you can use other metrics like entropy)
        weights = torch.stack([max1, max2], dim=1)
        weights = weights / weights.sum(dim=1, keepdim=True)
        # Weighted fusion of softmax probabilities
        dist = (dist1 * weights[:, 0].unsqueeze(1) +
                dist2 * weights[:, 1].unsqueeze(1))
    else:
        raise ValueError('Unknown metric')
    return dist, z_proto, zq
