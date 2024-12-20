import torch
import numpy as np


def hyperedge_concat(*H_list):
    """
    Concatenate hyperedge group in H_list
    :param H_list: Hyperedge groups which contain two or more hypergraph incidence matrix
    :return: Fused hypergraph incidence matrix
    """
    H = None
    for h in H_list:
        if h is not None and h != []:
            # for the first H appended to fused hypergraph incidence matrix
            if H is None:
                H = h
            else:
                if type(h) != list:
                    H = np.hstack((H, h))
                else:
                    tmp = []
                    for a, b in zip(H, h):
                        tmp.append(np.hstack((a, b)))
                    H = tmp
    return H


def generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H))
        return G


def _generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H, B X N_object X N_hyperedge
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    
    n_edge = H.shape[2]
    # the weight of the hyperedge
    W = torch.ones(n_edge, device=H.device)
    # the degree of the node
    DV = torch.sum(H * W, dim=2)
    # the degree of the hyperedge
    DE = torch.sum(H, dim=1)

    invDE = torch.diag_embed(torch.pow(DE, -1))
    DV2 = torch.diag_embed(torch.pow(DV, -0.5))
    W = torch.diag(W)
    HT = H.transpose(-1, -2)

    G = DV2.matmul(H).matmul(W).matmul(invDE).matmul(HT).matmul(DV2)
    return G
