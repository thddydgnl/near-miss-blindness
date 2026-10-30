"""Small ViT latent predictor over frozen DINOv2 patch tokens (3 history frames -> next frame tokens), R2-2 design
with configurable widths for the R2-11 crushing experiment.

  d   : residual-stream width (d_model)      -> "residual-width crush" sets d = mlp = width
  mlp : MLP hidden width (the neuron layer)  -> "MLP-hidden-only crush" keeps d = 256 and sets mlp = width
  nh  : attention heads, default max(1, d // 32)
Action pathway: zero-initialised AdaLN (as in R2-2):
    h = LN(x) * (1 + gamma(a)) + beta(a);   x = x + (1 + alpha(a)) * Sublayer(h)
Prediction: z_{t+1} = z_t + head(LN(x_current_frame_tokens)) with a zero-initialised head (init = copy model).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.nh, D // self.nh).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, N, D))


class Block(nn.Module):
    def __init__(self, d, nh, mlp):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = Attn(d, nh)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, mlp), nn.GELU(), nn.Linear(mlp, d))   # mlp[1] = the neuron layer

    def forward(self, x, mod=None):
        if mod is None:
            x = x + self.attn(self.ln1(x))
            return x + self.mlp(self.ln2(x))
        b1, g1, a1, b2, g2, a2 = mod.unsqueeze(2).unbind(1)          # each [B,1,d]
        x = x + (1 + a1) * self.attn(self.ln1(x) * (1 + g1) + b1)
        return x + (1 + a2) * self.mlp(self.ln2(x) * (1 + g2) + b2)


class Predictor(nn.Module):
    def __init__(self, d_in=384, d=256, L=6, nh=None, mlp=None, n_tok=64, n_hist=3):
        super().__init__()
        nh = nh or max(1, d // 32)
        mlp = mlp or 4 * d
        self.n_tok, self.n_hist, self.d, self.mlp_width = n_tok, n_hist, d, mlp
        self.inp = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.randn(1, 1, n_tok, d) * 0.02)
        self.tim = nn.Parameter(torch.randn(1, n_hist, 1, d) * 0.02)
        self.blocks = nn.ModuleList([Block(d, nh, mlp) for _ in range(L)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, d_in)
        # action pathway
        self.act_emb = nn.Sequential(nn.Linear(2, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())
        self.act_mod = nn.ModuleList([nn.Linear(d, 6 * d) for _ in range(L)])
        self.act_fin = nn.Linear(d, 2 * d)
        self.apply(self._init)
        for m in [self.head, self.act_fin, *self.act_mod]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

    def forward(self, H, a):
        """H: [B,3,64,384] standardised features of frames t-2..t; a: [B,2]. Returns predicted frame t+1 [B,64,384]."""
        B = H.shape[0]
        x = (self.inp(H) + self.pos + self.tim).reshape(B, self.n_hist * self.n_tok, self.d)
        e = self.act_emb(a)
        for i, blk in enumerate(self.blocks):
            x = blk(x, self.act_mod[i](e).view(B, 6, self.d))
        x = self.ln_f(x[:, -self.n_tok:])
        sh, sc = self.act_fin(e).view(B, 2, 1, self.d).unbind(1)
        x = x * (1 + sc) + sh
        return H[:, -1] + self.head(x)


# the crushing grid: name -> (d_model, mlp_hidden)
CONFIGS = {
    'w256': (256, 256),                      # shared wide reference (d = 256 for both crush types)
    'mlp64': (256, 64), 'mlp16': (256, 16), 'mlp8': (256, 8),     # MLP-hidden-only crush
    'res64': (64, 64), 'res16': (16, 16), 'res8': (8, 8),         # residual-width crush (stream and MLP = d)
}


def make_model(name):
    d, mlp = CONFIGS[name]
    return Predictor(d=d, mlp=mlp)


def n_params(m):
    return sum(p.numel() for p in m.parameters())
