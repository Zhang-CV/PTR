import torch
from einops import repeat
from torch import nn, einsum


def exists(val):
    return val is not None


def max_value(t):
    return torch.finfo(t.dtype).max


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)

# classes


class PointTransformerLayer(nn.Module):
    def __init__(self, args, d_model=3, pos_mlp_hidden_dim=64, num_neighbors=20):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.d_model = args.emb_dims
        self.h = args.heads
        self.to_qkv = nn.Linear(d_model, self.d_model * 3, bias=False)

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, pos_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(pos_mlp_hidden_dim, self.d_model)
        )

        self.attn_mlp = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.h),
            nn.ReLU(),
            nn.Linear(self.d_model * self.h, self.d_model),
        )

    def forward(self, x, pos):
        n, num_neighbors = x.shape[1], self.num_neighbors

        # get queries, keys, values
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        # calculate relative positional embeddings
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos_emb = self.pos_mlp(rel_pos)

        qk_rel = q[:, :, None, :] - k[:, None, :, :]

        # expand values
        v = repeat(v, 'b j d -> b i j d', i=n)

        # determine k nearest neighbors for each point, if specified
        if exists(num_neighbors) and num_neighbors < n:
            rel_dist = rel_pos.norm(dim=-1)
            dist, indices = rel_dist.topk(num_neighbors, largest=False)

            v = batched_index_select(v, indices, dim=2)
            qk_rel = batched_index_select(qk_rel, indices, dim=2)
            rel_pos_emb = batched_index_select(rel_pos_emb, indices, dim=2)

        # add relative positional embeddings to value
        v = v + rel_pos_emb

        # use attention mlp, making sure to add relative positional embedding first
        sim = self.attn_mlp(qk_rel + rel_pos_emb)

        # attention
        attn = sim.softmax(dim=-2)

        # aggregate
        agg = einsum('b i j d, b i j d -> b i d', attn, v)
        return agg
