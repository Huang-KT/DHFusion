import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, hw=256):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(hw, d_model)
        position = torch.arange(0, hw, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)  # [h*w, 1, c]

    def forward(self, x):  # [h*w, b, c]
        x = x + self.pe.to(dtype=x.dtype, device=x.device)
        return x


class EAM(nn.Module):

    def __init__(self, ft_in_ch, num_class, d_model,
                 nhead, dim_feedforward=2048, dropout=0.1):
        super(EAM, self).__init__()

        if ft_in_ch == d_model:
            self.align = nn.Identity()
        else:
            self.align = nn.Sequential(
                nn.Conv2d(ft_in_ch, d_model, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(d_model),
                nn.PReLU(),
                nn.Conv2d(d_model, d_model, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(d_model)
            )

        self.pos_encoding = PositionalEncoding(d_model=d_model, hw=256)  # nn.Identity()

        ## self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout) if dropout > 1e-5 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)
        
        ## ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = nn.PReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 1e-5 else nn.Identity()
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout) if dropout > 1e-5 else nn.Identity()

        ## Classifier
        self.hm_cls = nn.Linear(d_model, num_class)

        ## Get HeatMap
        self.get_hm = nn.Sequential(
            nn.Conv2d(d_model, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.Conv2d(48, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


    def forward(self, x: torch.Tensor):  # x: [b, 512, h, w]
        
        ft = x.clone()
        ft = self.align(ft)

        b, c, h, w = ft.shape
        ft = ft.flatten(2).permute(2, 0, 1)  # hw, b, c

        ## Self-Attention
        q = k = self.pos_encoding(ft)
        ft_self = self.self_attn(q, k, value=ft)[0]
        ft = ft + self.dropout1(ft_self)
        ft = ft_self
        ft = self.norm1(ft)

        ## FFN
        ft_ffn = self.linear2(self.dropout(self.activation(self.linear1(ft))))
        ft = ft + self.dropout3(ft_ffn)
        ft = self.norm3(ft_ffn)

        ft = ft.permute(1, 2, 0).view(b, c, h, w)

        ## Get HeatMap
        hm = self.get_hm(ft)
        
        ## Classification
        ft_cls = F.adaptive_avg_pool2d(ft, 1).squeeze()
        hm_cls = self.hm_cls(ft_cls)
        if len(hm_cls.shape) == 1:
            hm_cls = hm_cls.unsqueeze(0)

        out = x * hm + x

        return out, [hm, hm_cls]
