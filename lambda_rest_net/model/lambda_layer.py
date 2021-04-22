import torch
from torch import nn, einsum
from einops import rearrange

class LambdaLayer(nn.Module):
    def __init__(self, dim, *, dim_k, n = None, r = None, heads = 4, dim_out = None, dim_u = 1):
        super(LambdaLayer, self).__init__()
        print('t', dim_out , heads)
        assert (dim_out % heads) == 0, 'values dimension must be divisible by number of heads for multi-head query'
        assert (r % 2) == 1, 'Receptive kernel size should be odd'

        dim_out = dim_out if dim_out is not None else dim
        self.u = dim_u
        self.heads = heads

        dim_v = dim_out // heads

        self.to_q = nn.Conv2d(dim, dim_k * heads, 1, bias = False)
        self.to_k = nn.Conv2d(dim, dim_k * dim_u, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim_v * dim_u, 1, bias = False)

        self.norm_q = nn.BatchNorm2d(dim_k * heads)
        self.norm_v = nn.BatchNorm2d(dim_v * dim_u)

        self.local_contexts = True
        self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, r, r), padding = (0, r // 2, r // 2))


    def forward(self, x):
        b, c, hh, ww, u, h = *x.shape, self.u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b (h k) hh ww -> b h k (hh ww)', h = h)
        k = rearrange(k, 'b (u k) hh ww -> b u k (hh ww)', u = u)
        v = rearrange(v, 'b (u v) hh ww -> b u v (hh ww)', u = u)

        k = k.softmax(dim=-1)

        λc = einsum('b u k m, b u v m -> b k v', k, v)
        Yc = einsum('b h k n, b k v -> b h v n', q, λc)

        v = rearrange(v, 'b u v (hh ww) -> b u v hh ww', hh = hh, ww = ww)
        λp = self.pos_conv(v)
        Yp = einsum('b h k n, b k v n -> b h v n', q, λp.flatten(3))

        Y = Yc + Yp
        out = rearrange(Y, 'b h v (hh ww) -> b (h v) hh ww', hh = hh, ww = ww)
        return out