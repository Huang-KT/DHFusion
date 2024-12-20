import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time

from .dis_utils import Eu_dist, Cos_dist
from .hypergraph_utils import hyperedge_concat


def construct_H_with_KNN_from_distance(dist, k_neig, dis_type, is_probH=True, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix, B X N_obj X N_obj
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: B X N_object X N_hyperedge
    """

    B, n_obj = dist.shape[:2]
    n_edge = n_obj
    center_idx = torch.arange(0, n_obj)
    dist[:, center_idx, center_idx] = 0

    ## Get the index of K-Nearest-Neighbor
    _, nearest_indices = torch.topk(dist, k=k_neig, dim=-1, largest=False)  # [B, N_obj, k_neig]

    idx = torch.zeros((B, n_obj, n_edge), device=dist.device)
    idx[torch.arange(B).view(B, 1, 1), torch.arange(n_obj).view(n_obj, 1), nearest_indices] = 1.0

    avg_dis = torch.mean(dist, dim=-1, keepdim=True)  # [B, N_obj]

    #################################
    if dis_type == "Cosine":
        threshold = 0.5
        idx[dist > threshold] = 0.0
    #################################

    H = None
    if is_probH:
        H = torch.exp(-dist ** 2 / (m_prob * avg_dis) ** 2) * idx
    else:
        H = idx

    return H.transpose(-1, -2)


# K-Nearest Neighbor
def construct_H_with_KNN(X, K_neigs=[10], split_diff_scale=False, is_probH=True, m_prob=1, dis_type="Cosine") -> torch.Tensor:
    """
    :param X: [b, c, H, W]
    :param K_neigs: the number of neighbor expansion
    :param split_diff_scale: whether split hyperedge group at different neighbor scale
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: [b, H*W, N_hyperedge]
    """
    with torch.no_grad():
        if len(X.shape) == 4:
            b, c, H, W = X.shape
            X = X.reshape(b, c, H * W)

        if type(K_neigs) == int:
            K_neigs = [K_neigs]

        if dis_type == "Euclidean":
            dist = Eu_dist(X)
        elif dis_type == "Cosine":
            dist = Cos_dist(X)
        else:
            assert False
        
        H = []
        for k_neig in K_neigs:
            H_tmp = construct_H_with_KNN_from_distance(dist, k_neig, dis_type, is_probH, m_prob)
            if not split_diff_scale:
                H = hyperedge_concat(H, H_tmp)
            else:
                H.append(H_tmp)
    return H


# H = phi * Lambda * phi.T * Omega
class LearningH(nn.Module):
    def __init__(self, in_features, out_features, features_height, features_width,
                 edges, filters, apply_bias=True, epsilon=0.0):
        super(LearningH, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.features_height = features_height
        self.features_width = features_width
        self.vertices = self.features_height * self.features_width
        self.edges = edges
        self.apply_bias = apply_bias
        self.filters = filters
        self.epsilon = epsilon

        self.phi_conv = nn.Conv2d(self.in_features, self.filters, kernel_size=1, stride=1, padding=0)
        init.xavier_normal_(self.phi_conv.weight)
        self.A_conv = nn.Conv2d(self.in_features, self.filters, kernel_size=1, stride=1, padding=0)
        init.xavier_normal_(self.A_conv.weight)
        self.M_conv = nn.Conv2d(self.in_features, self.edges, kernel_size=7, stride=1, padding=3)
        init.xavier_normal_(self.M_conv.weight)

    def forward(self, x:torch.Tensor):
        # Phi Matrix
        phi = self.phi_conv(x)
        phi = phi.view(-1, self.vertices, self.filters)

        # Lambda Matrix
        A = F.adaptive_avg_pool2d(x, (1, 1))
        A = self.A_conv(A)
        A = torch.diag_embed(A.view(-1, self.filters))
        
        # Omega Matrix
        M = self.M_conv(x)
        M = M.view(-1, self.vertices, self.edges)

        # Incidence matrix
        H = phi.matmul(A).matmul(phi.transpose(1, 2)).matmul(M)

        ### Softmax
        H = F.softmax(H, dim=-1)

        if self.epsilon > 1e-5:
            torch.where(H < self.epsilon, torch.tensor(0.0, dtype=H.dtype, device=H.device), H)

        return H
