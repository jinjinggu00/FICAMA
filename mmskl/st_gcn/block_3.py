# 只有3个block的gcn，用于实现painet的早期融合
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from . import ConvTemporalGraphical, Graph

import numpy as np


def zero(x):
    return 0


def iden(x):
    return x


class GCN_3(nn.Module):
    r"""Spatial temporal graph convolutional networks.

    Args:
          - in_channels: 输入数据通道数
          - num_class: 分类数
          - graph_cfg: 构建图的参数
          - edge_importance_weighting: 是否使用可学习的边重要性
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph
        **kwargs (optional): 图卷积的其它参数

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    输入形状:
      - x: (N, in_channels, T, V, M)
      - N: batch大小
      - T: 输入序列长度
      - V: 图节点数
      - M: 每帧实例数
    输出形状:
      - (N, num_class)
    """

    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 edge_importance_weighting=True,
                 data_bn=True,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = torch.tensor(self.graph.A,
                         dtype=torch.float32,
                         requires_grad=False)
        self.register_buffer('A', A)

        # build networks
        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        # 输入数据批标准化
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn_block(in_channels,
        #                  64,
        #                  kernel_size,
        #                  1,
        #                  residual=False,
        #                  **kwargs0),
        #     st_gcn_block(64, 64, kernel_size, 2, **kwargs),
        #     st_gcn_block(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn_block(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn_block(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn_block(256, 256, kernel_size, 2, **kwargs),
        #     st_gcn_block(256, 64, kernel_size, 1, **kwargs),
        # ))
        # self.st_gcn_networks = nn.ModuleList((
        #     st_gcn_block(in_channels,
        #                  64,
        #                  kernel_size,
        #                  1,
        #                  residual=False,
        #                  **kwargs0),
        #     st_gcn_block(64, 64, kernel_size, 1, **kwargs),
        #     st_gcn_block(64, 128, kernel_size, 2, **kwargs),
        #     st_gcn_block(128, 128, kernel_size, 1, **kwargs),
        #     st_gcn_block(128, 256, kernel_size, 2, **kwargs),
        #     st_gcn_block(256, 256, kernel_size, 1, **kwargs),
        #     st_gcn_block(256, 64, kernel_size, 1, **kwargs),
        # ))
        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels,
                         64,
                         kernel_size,
                         1,
                         residual=False,
                         **kwargs0),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
            st_gcn_block(64, 64, kernel_size, 1, **kwargs),
        ))

        # initialize parameters for edge importance weighting边重要性参数初始化
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # fcn for prediction
        # self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    # 正向传播
    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forward
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        # N channel 50 25

        # global pooling x.size()[2:] = (300, 25)
        # x = F.avg_pool2d(x, x.size()[2:])
        # 全局池化
        NM, C, T, V = x.size()
        x = x.view(N, M, -1, T, V).permute(0, 2, 3, 4, 1)

        # prediction
        # x = self.fcn(x)
        # x = x.view(x.size(0), -1)

        return x

    def extract_feature(self, x):

        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        # forwad
        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        # prediction
        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)

        return output, feature


class st_gcn_block(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
          - in_channels: 输入通道数
          - out_channels: 输出通道数
          - kernel_size: 时域和图卷积核大小
          - stride: 时域卷积步长
          - dropout: Dropout比率
          - residual: 是否使用残差连接
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.

    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        # 残差连接
        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        return self.relu(x), A
