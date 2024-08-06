import math

import torch
from torch import nn
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        # pe is of shape max_len(5000) x 2048(last layer of FC)
        pe = torch.zeros(max_len, d_model)
        # position is of shape 5000 x 1
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        # pe contains a vector of shape 1 x 5000 x 2048
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class RotationPositionalEncoding(nn.Module):
    """Implement the Rotation PE function."""

    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(RotationPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model // 4, 1) * -(math.log(10000.0) / (d_model // 4)))
        pe[:, 0::4] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::4] = torch.cos(position * div_term) * self.pe_scale_factor
        pe[:, 2::4] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 3::4] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Implement the Learned PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(LearnedPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Token_Perceptron(torch.nn.Module):
    '''
        2-layer Token MLP
    '''

    def __init__(self, in_dim):
        super(Token_Perceptron, self).__init__()
        # in_dim 8
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Applying the linear layer on the input
        output = self.inp_fc(x)  # B x 2048 x 8

        # Apply the relu non-linearity
        output = self.relu(output)  # B x 2048 x 8

        # Apply the 2nd linear layer
        output = self.out_fc(output)

        return output


class Bottleneck_Perceptron_2_layer(torch.nn.Module):
    '''
        2-layer Bottleneck MLP
    '''

    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_2_layer, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.out_fc(output)

        return output


class Bottleneck_Perceptron_3_layer_res(torch.nn.Module):
    '''
        3-layer Bottleneck MLP followed by a residual layer
    '''

    def __init__(self, in_dim):
        # in_dim 2048
        super(Bottleneck_Perceptron_3_layer_res, self).__init__()
        self.inp_fc = nn.Linear(in_dim, in_dim // 2)
        self.hid_fc = nn.Linear(in_dim // 2, in_dim // 2)
        self.out_fc = nn.Linear(in_dim // 2, in_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        output = self.relu(self.inp_fc(x))
        output = self.relu(self.hid_fc(output))
        output = self.out_fc(output)

        return output + x  # Residual output


class MLP_Mix_base(nn.Module):
    """
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """

    def __init__(self, in_dim, seq_len):
        super(MLP_Mix_base, self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len)  # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)

        max_len = int(seq_len * 1.5)  # seq_len = 8
        self.pe = PositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature
        """

        # Add a position embedding to the 8 frames
        x = self.pe(x)  # B x 8 x 2048

        # Store the residual for use later
        residual1 = x  # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1  # B x 8 x 2048

        # Storing a residual
        residual2 = out  # B x 8 x 2048

        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2  # B x 8 x 2048

        return out


class MLP_Mix_Enrich(nn.Module):
    """
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """

    def __init__(self, in_dim, seq_len, pe_type='positional'):
        super(MLP_Mix_Enrich, self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len)  # seq_len = 8 frames
        # self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)
        self.pe_type = pe_type
        max_len = int(seq_len * 1.5)
        if pe_type == 'positional':
            self.pe = PositionalEncoding(in_dim, 0.1, max_len)
        elif pe_type == 'rotation':
            self.pe = RotationPositionalEncoding(in_dim, 0.1, max_len)
        elif pe_type == 'learned':
            self.pe = LearnedPositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W ) # B(25/20) x 8 x 2048
            returns :
                out : self MLP-enriched value + input feature
        """

        # Add a position embedding to the 8 frames
        if self.pe_type != 'none':
            x = self.pe(x)  # B x 8 x 2048

        # Store the residual for use later
        residual1 = x  # B x 8 x 2048

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 2048 x 8
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1  # B x 8 x 2048

        # # Storing a residual
        # residual2 = out  # B x 8 x 2048
        #
        # # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        # out = self.Bot_MLP(out) + residual2  # B x 8 x 2048

        return out


class MLP_Mix_Joint(nn.Module):
    """
        Pure Token-Bottleneck MLP-based enriching features mechanism
    """

    def __init__(self, in_dim, seq_len, pe_type='positional'):
        super(MLP_Mix_Joint, self).__init__()
        # in_dim = 2048
        self.Tok_MLP = Token_Perceptron(seq_len)  # seq_len = 8 frames
        self.Bot_MLP = Bottleneck_Perceptron_2_layer(in_dim)
        self.pe_type = pe_type
        max_len = int(seq_len * 1.5)  # seq_len = 8
        if pe_type == 'positional':
            self.pe = PositionalEncoding(in_dim, 0.1, max_len)
        elif pe_type == 'rotation':
            self.pe = RotationPositionalEncoding(in_dim, 0.1, max_len)
        elif pe_type == 'learned':
            self.pe = LearnedPositionalEncoding(in_dim, 0.1, max_len)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B*t X V X C ) # B(50/5/25) x V(25/18) x 256
            returns :
                out : self MLP-enriched value + input feature
        """

        # Add a position embedding to the 8 frames
        if self.pe_type != 'none':
            x = self.pe(x)  # B x 25 x 256

        # Store the residual for use later
        residual1 = x  # B x 25 x 256

        # Pass it via a 2-layer Token MLP followed by Residual Layer
        # Permuted before passing into the MLP: B x 256 x 25
        out = self.Tok_MLP(x.permute(0, 2, 1)).permute(0, 2, 1) + residual1  # B x 25 x 256

        # Storing a residual
        residual2 = out  # B x 8 x 2048

        # Pass it via 2-layer Bottleneck MLP defined on Channel(2048) features
        out = self.Bot_MLP(out) + residual2  # B x 8 x 2048

        return out
