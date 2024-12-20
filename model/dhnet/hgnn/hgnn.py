import torch
from torch import nn
from timm.models.layers import DropPath


class Apply_HypergraphConv(nn.Module):
    def __init__(self, in_features, out_features, drop_path=0.0, apply_bias=True):
        super(Apply_HypergraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.apply_bias = apply_bias
        self.weight_2 = nn.Parameter(torch.Tensor(self.in_features, self.out_features))
        nn.init.xavier_normal_(self.weight_2)

        if apply_bias:
            self.bias_2 = nn.Parameter(torch.Tensor(1, self.out_features))
            nn.init.xavier_normal_(self.bias_2)
        else:
            self.register_parameter('bias_2', None)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, HyperG: torch.Tensor):
        
        _, _, h, w = x.shape
        
        features = x.contiguous().view(-1, h*w, self.in_features)

        # Hypergraph Convolution
        out = HyperG.matmul(features)
        out = torch.matmul(out, self.weight_2)

        if self.apply_bias:
            out = out + self.bias_2

        out = out.view(-1, self.out_features, h, w)

        return self.drop_path(out) + x
