import math
import time

import open3d as o3d
import torch
import torch.nn as nn
import torch.nn.functional as F

from point_transformer import PointTransformerLayer


class PointTransformer(nn.Module):
    def __init__(self, args):
        super(PointTransformer, self).__init__()
        self.linear1 = nn.Linear(3, args.emb_dims)
        self.linear2 = nn.Linear(args.emb_dims, args.emb_dims)
        self.linear3 = nn.Linear(args.emb_dims, args.emb_dims)
        self.PointTransformerLayer1 = PointTransformerLayer(args, d_model=args.emb_dims, pos_mlp_hidden_dim=8, num_neighbors=20)

    def forward(self, x, pos):
        x = self.linear1(x)
        x1 = self.linear2(x)
        x1 = self.PointTransformerLayer1(x1, pos)
        x = x + self.linear3(x1)
        return x


def attention(query, key, value, dropout):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    p_attn = torch.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    output = torch.matmul(p_attn, value)
    return output


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):   # h, d_model : dims of input & output

        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(4))
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous() \
                             for l, x in zip(self.linears, (query, key, value))]
        # embedding  #(batch_size, h, num_points, d_k)
        x = attention(query, key, value, self.dropout)  # (batch_size, h, num_points, d_k)
        concat = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)  # concat attention
        out = self.linears[-1](concat)  # apply the last linear layer on output (batch_size, num_points, h * d_k)

        return out


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff=1024, dropout=0.1):
        super(FeedForward, self).__init__()

        self.feedforward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, out):
        out = self.feedforward(out)
        return out     # (batch_size, num_points, d_model)


class EncoderLayer(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attn = MultiHeadedAttention(h, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, d_ff=1024, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = F.layer_norm(x, x.size()[1:])
        x = x + self.dropout1(self.attn(x2, x2, x2))
        x2 = F.layer_norm(x, x.size()[1:])
        x = x + self.dropout2(self.ff(x2))
        return x   # (batch_size, num_points, d_model)


class DecoderLayer(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(DecoderLayer, self).__init__()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.attn1 = MultiHeadedAttention(h, d_model, dropout=dropout)
        self.attn2 = MultiHeadedAttention(h, d_model, dropout=dropout)

        self.ff = FeedForward(d_model, d_ff=1024, dropout=dropout)

    def forward(self, x, l_outputs):
        x2 = F.layer_norm(x, x.size()[1:])
        x = x + self.dropout1(self.attn1(x2, x2, x2))
        x2 = F.layer_norm(x, x.size()[1:])
        x = x + self.dropout2(self.attn2(x2, l_outputs, l_outputs))
        x2 = F.layer_norm(x, x.size()[1:])
        x = x + self.dropout3(self.ff(x2))

        return x  # (batch_size, num_points, d_model)


class Encoder(nn.Module):
    def __init__(self, h, d_model, n, dropout=0.1):
        super(Encoder, self).__init__()
        self.n = n  # the num of the EncoderLayers
        self.layers = nn.ModuleList(EncoderLayer(h, d_model, dropout) for _ in range(n))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = F.layer_norm(x, x.size()[1:])
        return x


class Decoder(nn.Module):
    def __init__(self, h, d_model, n, dropout=0.1):
        super(Decoder, self).__init__()
        self.n = n
        self.layers = nn.ModuleList(DecoderLayer(h, d_model, dropout=dropout) for _ in range(n))

    def forward(self, x, l_outputs):
        for layer in self.layers:
            x = layer(x, l_outputs)
        x = F.layer_norm(x, x.size()[1:])
        return x


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.h = args.heads
        self.d_model = args.emb_dims
        self.n = args.n
        self.ff_dims = args.ff_dims
        self.dropout = args.dropout
        self.encoder = Encoder(self.h, self.d_model, self.n, self.dropout)
        self.decoder = Decoder(self.h, self.d_model, self.n, self.dropout)

    def forward(self, src, target):
        l_output = self.encoder(src)
        r_output = self.decoder(target, l_output)
        return r_output  # (batch_size, num_points, d_model)


class SVDHead(nn.Module):
    def __init__(self):
        super(SVDHead, self).__init__()
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1

    def forward(self, src_embedding, target_embedding, src, target):
        # src_embedding : (batch_size, num_points, feature_dims)
        d_k = src_embedding.size(2)
        scores = torch.matmul(src_embedding, target_embedding.transpose(2, 1).contiguous()) / math.sqrt(d_k)
        scores = torch.softmax(scores, dim=2)   # (batch_size, target_num_points, src_num_points)
        tgt_corr = torch.matmul(scores, src)
        tgt_centered = target - target.mean(dim=1, keepdim=True)
        tgt_corr_centered = tgt_corr - tgt_corr.mean(dim=1, keepdim=True)
        # SVD decomposition
        H = torch.matmul(tgt_centered.transpose(2, 1).contiguous(), tgt_corr_centered)

        U, _, V = torch.svd(H)
        U = U
        V = V
        C = torch.eye(3).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
        C[:, 2, 2] = torch.det(U @ V.transpose(1, 2))
        R = U @ C @ V.transpose(1, 2)
        t = target.mean(dim=1, keepdim=True).transpose(1, 2) - R @ tgt_corr.mean(dim=1, keepdim=True).transpose(1, 2)
        return R, t.transpose(1, 2)


def ICP(src, target):
    src_temp = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(src.squeeze(0).detach().cpu()))
    target_temp = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(target.squeeze(0).detach().cpu()))
    threshold = 0.1
    trans_init = torch.eye(4, 4)
    start = time.time()
    reg_p2p = o3d.pipelines.registration.registration_icp(
        target_temp, src_temp, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    end = time.time()
    transformation = reg_p2p.transformation
    rotation_ab_icp = torch.tensor(transformation[0: 3][:, 0: 3]).unsqueeze(0).cuda().float()
    translation_ba_icp = torch.tensor(transformation[0: 3][:, 3:4]).unsqueeze(0).cuda().float().transpose(2, 1)
    translation_ab_icp = -torch.bmm(translation_ba_icp, rotation_ab_icp)
    return rotation_ab_icp, translation_ab_icp, end - start


class PTR(nn.Module):
    def __init__(self, args):
        super(PTR, self).__init__()
        self.emb_network = PointTransformer(args=args)
        self.transformer = Transformer(args=args)
        self.svd = SVDHead()
        self.icp = args.icp

    def forward(self, src, target):
        start = time.time()
        target_embedding = self.emb_network(target, target)
        src_embedding = self.emb_network(src, src)
        src_embedding_p = self.transformer(src_embedding, target_embedding)
        target_embedding_p = self.transformer(target_embedding, src_embedding)
        rotation_ab, translation_ab = self.svd(src_embedding_p, target_embedding_p, src, target)

        src = torch.bmm(rotation_ab, src.transpose(2, 1)) + translation_ab.transpose(2, 1)
        src = src.transpose(2, 1).detach()

        end = time.time()
        time_cost_dcp = end - start
        if self.icp:
            rotation_ab_icp, translation_ab_icp, _ = ICP(target, src)
            rotation_ab = torch.bmm(rotation_ab_icp, rotation_ab)
            translation_ab = torch.bmm(rotation_ab_icp, translation_ab.transpose(2, 1)) + translation_ab_icp.transpose(2, 1)
            translation_ab = translation_ab.transpose(2, 1)
        return rotation_ab, translation_ab, time_cost_dcp


