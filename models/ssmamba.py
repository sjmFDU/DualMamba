import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from timm.models.layers import DropPath, to_2tuple

from .pos_embed import get_2d_sincos_pos_embed
import copy
import math
from typing import Optional, Callable
from einops import rearrange, repeat

from fvcore.nn import flop_count, parameter_count
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except:
    pass


def selective_scan_flop_jit(inputs, outputs):
    # xs, dts, As, Bs, Cs, Ds (skip), z (skip), dt_projs_bias (skip)
    assert inputs[0].debugName().startswith("xs") # (B, D, L)
    assert inputs[2].debugName().startswith("As") # (D, N)
    assert inputs[3].debugName().startswith("Bs") # (D, N)
    with_Group = len(inputs[3].type().sizes()) == 4
    with_D = inputs[5].debugName().startswith("Ds")
    if not with_D:
        with_z = inputs[5].debugName().startswith("z")
    else:
        with_z = inputs[6].debugName().startswith("z")
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_ref(B=B, L=L, D=D, N=N, with_D=with_D, with_Z=with_z, with_Group=with_Group)
    return flops

def flops_selective_scan_ref(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_Group=True, with_complex=False):
    """
    u: r(B D L)
    delta: r(B D L)
    A: r(D N)
    B: r(B N L)
    C: r(B N L)
    D: r(D)
    z: r(B D L)
    delta_bias: r(D), fp32

    ignores:
        [.float(), +, .softplus, .shape, new_zeros, repeat, stack, to(dtype), silu]
    """
    import numpy as np

    # fvcore.nn.jit_handles
    def get_flops_einsum(input_shapes, equation):
        np_arrs = [np.zeros(s) for s in input_shapes]
        optim = np.einsum_path(equation, *np_arrs, optimize="optimal")[1]
        for line in optim.split("\n"):
            if "optimized flop" in line.lower():
                # divided by 2 because we count MAC (multiply-add counted as one flop)
                flop = float(np.floor(float(line.split(":")[-1]) / 2))
                return flop


    assert not with_complex

    flops = 0 # below code flops = 0
    if False:
        ...
        """
        dtype_in = u.dtype
        u = u.float()
        delta = delta.float()
        if delta_bias is not None:
            delta = delta + delta_bias[..., None].float()
        if delta_softplus:
            delta = F.softplus(delta)
        batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
        is_variable_B = B.dim() >= 3
        is_variable_C = C.dim() >= 3
        if A.is_complex():
            if is_variable_B:
                B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
            if is_variable_C:
                C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
        else:
            B = B.float()
            C = C.float()
        x = A.new_zeros((batch, dim, dstate))
        ys = []
        """

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
    if False:
        ...
        """
        deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
        if not is_variable_B:
            deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
        else:
            if B.dim() == 3:
                deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
            else:
                B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
        if is_variable_C and C.dim() == 4:
            C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
        last_state = None
        """

    in_for_flops = B * D * N
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops
    if False:
        ...
        """
        for i in range(u.shape[2]):
            x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
            if not is_variable_C:
                y = torch.einsum('bdn,dn->bd', x, C)
            else:
                if C.dim() == 3:
                    y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                else:
                    y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
            if i == u.shape[2] - 1:
                last_state = x
            if y.is_complex():
                y = y.real * 2
            ys.append(y)
        y = torch.stack(ys, dim=2) # (batch dim L)
        """

    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L
    if False:
        ...
        """
        out = y if D is None else y + u * rearrange(D, "d -> d 1")
        if z is not None:
            out = out * F.silu(z)
        out = out.to(dtype=dtype_in)
        """

    return flops

class SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            bias = False,
            gaussian = False,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank


        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # x proj; dt proj ============================
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.D = self.D_init(self.d_inner)  # (D)


        self.selective_scan = selective_scan_fn


        # out norm ===================================
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, L, d = x.shape
        x = x.permute(0, 2, 1)
        
        
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=L)
        A = -torch.exp(self.A_log.float())  # (k * d, d_state)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=L).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=L).contiguous()

        y = self.selective_scan(
            x, dt,
            A, B, C, self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        # assert out_y.dtype == torch.float
        y = rearrange(y, "b d l -> b l d")
        y = self.out_norm(y)
        return y

    def forward(self, x: torch.Tensor):
        B, L, d = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        y = self.forward_core(x)
        y = y * F.silu(z)
        out = self.out_proj(y)

        return out

class block_1D(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        attn_drop_rate: float = 0,
        d_state: int = 16,
        bi: bool = True,
        cls: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SSM(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.bi = bi
        self.cls = cls

    def forward(self, input: torch.Tensor):
        x = self.ln_1(input)
        x1 = self.self_attention(x)
        if self.bi:
            if self.cls:
                x2 = x[:,0:-1,:]
                cls_token = x[:,-1:,:]
                x2 = torch.flip(x2, dims=[1])
                x2 = torch.cat((x2, cls_token), dim=1)
                x3 = self.self_attention(x2)

                x2 = x3[:,0:-1,:]
                cls_token = x3[:,-1:,:]
                x3 = torch.flip(x2, dims=[1])
                x3 = torch.cat((x3, cls_token), dim=1)
            else:
                x3 = torch.flip(x, dims=[1])
                x3 = self.self_attention(x3)
                x3 = torch.flip(x3, dims=[1])
            return self.drop_path((x1+x3)/2) + input
        else:
            return self.drop_path(x1) + input

def positional_embedding_1d(seq_len, embed_size):
    position_enc = torch.zeros(seq_len, embed_size)
    position = torch.arange(0, seq_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2) * -(torch.log(torch.tensor(10000.0)) / embed_size))
    position_enc[:, 0::2] = torch.sin(position.float() * div_term)
    position_enc[:, 1::2] = torch.cos(position.float() * div_term)
    return position_enc.unsqueeze(0)

class spectral_spatial_block(nn.Module):
    def __init__(self, embed_dim, bi=False, N=8, drop_path=0.0, norm_layer=nn.LayerNorm, cls = True, fu = True):
        super(spectral_spatial_block, self).__init__()
        self.spa_block = block_1D(
            # This module uses roughly 3 * expand * d_model^2 parameters
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls,
            # gaussian = True
            )
        self.spe_block = block_1D(
            # This module uses roughly 3 * expand * d_model^2 parameters
            hidden_dim=embed_dim, # Model dimension d_model
            drop_path = drop_path,
            bi = bi,
            cls = cls
            )
        self.linear = nn.Linear(N, N)
        self.norm = norm_layer(embed_dim)
        self.l1= nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias = False),
            nn.Sigmoid(),)
        self.fu = fu
    def forward(self, x_spa, x_spe):
        ###  x:(B, L, D)
        x_spa = self.spa_block(x_spa)   #(N, HW/P^2, D)
        B, N, D = x_spa.shape
        x_spe = self.spe_block(x_spe)   #(N, B, D)
        _,N1,_ = x_spe.shape

        if self.fu:
            x_spa_c = x_spa[:,(N-1)//2,:]
            x_spe_c = x_spe.mean(1)
            sig = self.l1((x_spa_c+x_spe_c)/2).unsqueeze(1)
            x_spa = x_spa*sig.expand(-1,N,-1)
            x_spe = x_spe*sig.expand(-1,N1,-1)

        return x_spa, x_spe



class PatchEmbed_1D(nn.Module):
    """ 1D signal to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=2, in_chans=1, embed_dim=64, norm_layer=None):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        x = x.transpose(1, 2)  # BCL -> BLC
        x = self.norm(x)
        return x

class PatchEmbed_2D(nn.Module):
    """ 2D image to Patch Embedding
    """
    def __init__(self, img_size=(224,224), patch_size=16, in_chans=3, embed_dim=64, norm_layer=None, flatten = True):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        if self.flatten:
            # x = spiral_flatten(x).transpose(1, 2)
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
            # x = s_flatten(x).transpose(1, 2)
        x = self.norm(x)
        return x

class PatchEmbed_Spe(nn.Module):
    """ 2D image to Patch Embedding
    """
    def __init__(self, img_size=(9,9), patch_size=2, embed_dim=64, norm_layer=None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size

        self.proj = nn.Conv1d(
            in_channels=img_size[0]*img_size[1],
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C, -1)
        x = x.transpose(2,1)
        x = self.proj(x)
        x = x.transpose(2,1)
        x = self.norm(x)
        return x

class mamba_1D_model(nn.Module):
    """
    从（27 27）中中心裁取（9 9）的patch，然后通过1D卷积提取特征
    """
    def __init__(self, img_size=(3,3), spa_img_size=(224, 224), nband=107, patch_size=1, embed_dim=128, nclass=10, drop_path=0.0, depth=4, bi=True, norm_layer=nn.LayerNorm, global_pool=True, cls = True):
        super().__init__()
        self.patch_embed = PatchEmbed_Spe(img_size, patch_size, embed_dim, norm_layer)
        if nband % patch_size == 0:
          self.num_patch = nband // patch_size
        else:
          self.num_patch = nband // patch_size 
        self.half_spe_patch_size = img_size[0] // 2
        self.half_spa_patch_size = spa_img_size[0] // 2
        self.cls = cls
        if self.cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cs =  -1
        else:
            self.cs = self.num_patch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([
                block_1D(hidden_dim=embed_dim, drop_path = drop_path, bi = bi, cls = cls) for i in range(depth)
            ])
        self.head = nn.Linear(embed_dim, nclass)
        self.pos_embed = nn.Parameter(positional_embedding_1d(self.num_patch+1, embed_dim), requires_grad = False)
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm

    def forward_features(self, x):
        x_spe = x[:,:,self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1,
                            self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1]
        x = self.patch_embed(x_spe)
        # append cls token
        x = x + self.pos_embed[:, :-1, :]
        if self.cls:
            cls_token = self.cls_token + self.pos_embed[:, -1:, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)      
            x = torch.cat((x, cls_tokens), dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, -1]
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class mamba_2D_model(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, hid_chans = 32, embed_dim=128, nclass=10, drop_path=0.0, depth=4, bi=True, norm_layer=nn.LayerNorm, global_pool=True, cls = True):
        super().__init__()

        self.dimen_redu = nn.Sequential(
                  nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
                  nn.BatchNorm2d(hid_chans),
                  nn.ReLU(),

                  nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
                  nn.BatchNorm2d(hid_chans),
                #   nn.ReLU(),
            )

        self.patch_embed = PatchEmbed_2D(img_size, patch_size, hid_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls = cls
        if self.cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.cs =  -1
        else:
            self.cs = num_patches
        self.blocks = nn.ModuleList([
                  block_1D(hidden_dim=embed_dim, drop_path = drop_path,bi = bi, cls = cls) for i in range(depth)
                          ])
        self.head = nn.Linear(embed_dim, nclass)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        self.initialize_weights()
    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def forward_features(self, x):
        x = self.dimen_redu(x)
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
            cls_token = self.cls_token + self.pos_embed[:, -1:, :]
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((x, cls_tokens), dim=1)
        for blk in self.blocks:
            x = blk(x)
        if self.global_pool:
            x = x[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, -1]
        return outcome

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class mamba_SS_model(nn.Module):
    def __init__(self, spa_img_size=27,spe_img_size=3, spa_patch_size=3, spe_patch_size=2, in_chans=200, hid_chans = 64, embed_dim=64, nclass=10, drop_path=0.0, depth=4, bi=True, 
                 norm_layer=nn.LayerNorm, global_pool=True, cls = True, fu=True):
        super().__init__()
        spa_img_size = (spa_img_size, spa_img_size)
        spe_img_size = (spe_img_size, spe_img_size)
        self.dimen_redu = nn.Sequential(
            nn.Conv2d(in_chans, hid_chans, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hid_chans),
            # nn.GroupNorm(4, hid_chans),
            nn.ReLU(),

            nn.Conv2d(hid_chans, hid_chans, 1, 1, 0, bias=True),
            nn.BatchNorm2d(hid_chans),
            # nn.GroupNorm(4, hid_chans),
            # nn.SiLU(),
            )

        self.half_spa_patch_size = spa_img_size[0] // 2
        self.half_spe_patch_size = spe_img_size[0] // 2
        self.spe_patch_embed = PatchEmbed_Spe(img_size=spe_img_size, patch_size=spe_patch_size, embed_dim=embed_dim)
        self.spa_patch_embed = PatchEmbed_2D(spa_img_size, spa_patch_size, hid_chans, embed_dim)
        spa_num_patches = self.spa_patch_embed.num_patches
        if in_chans % spe_patch_size ==0:
          spe_num_patches = in_chans//spe_patch_size
        else:
          spe_num_patches = in_chans//spe_patch_size

        self.cls = cls
        if self.cls:
          self.spa_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          self.spe_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
          N = spa_num_patches+spe_num_patches+2
          self.cs = -1
        else:
          N = spa_num_patches+spe_num_patches
          self.cs = N

        self.blocks = nn.ModuleList([
                spectral_spatial_block(embed_dim, bi, N=N, drop_path = drop_path, cls = self.cls, fu = fu) for i in range(depth)
                        ])
        self.head = nn.Linear(embed_dim, nclass)
        self.spa_pos_embed = nn.Parameter(torch.zeros(1, spa_num_patches+1, embed_dim), requires_grad=False)
        self.spe_pos_embed = nn.Parameter(positional_embedding_1d(spe_num_patches+1, embed_dim), requires_grad=False)

        self.norm = norm_layer(embed_dim)
        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)
            del self.norm
        self.initialize_weights()
        


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        spa_pos_embed = get_2d_sincos_pos_embed(self.spa_pos_embed.shape[-1], int(self.spa_patch_embed.num_patches**.5), cls_token=True)
        self.spa_pos_embed.data.copy_(torch.from_numpy(spa_pos_embed).float().unsqueeze(0))


    def forward_features(self, x):
        x_spa = self.dimen_redu(x)
        x_spa = self.spa_patch_embed(x_spa)
        x_spa = x_spa + self.spa_pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
          spa_cls_token = self.spa_cls_token + self.spa_pos_embed[:, -1:, :]
          spa_cls_tokens = spa_cls_token.expand(x_spa.shape[0], -1, -1)
          x_spa = torch.cat((x_spa, spa_cls_tokens), dim=1)

        x_spe = self.spe_patch_embed(x[:,:,self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1,
                                       self.half_spa_patch_size-self.half_spe_patch_size:self.half_spa_patch_size+self.half_spe_patch_size+1])
        x_spe = x_spe + self.spe_pos_embed[:, :-1, :]
        # append cls token
        if self.cls:
          spe_cls_token = self.spe_cls_token + self.spe_pos_embed[:, -1:, :]
          spe_cls_tokens = spe_cls_token.expand(x_spe.shape[0], -1, -1)
          x_spe = torch.cat((x_spe, spe_cls_tokens), dim=1)

        for blk in self.blocks:
            x_spa, x_spe = blk(x_spa, x_spe)
        if self.global_pool:
            x_spa = x_spa[:, 0:self.cs, :].mean(dim=1)  # global pool without cls token
            x_spe = x_spe[:, 0:self.cs, :].mean(dim=1)
            outcome = self.fc_norm((x_spa + x_spe)/2)
        else:
            outcome = (x_spa[:, -1] + x_spe[:, -1])/2
        return outcome

    def forward(self, x):
        x = x.squeeze(1)
        x = self.forward_features(x)
        out = self.head(x)  
            
        return out

    def flops(self, shape=(1, 48, 27, 27), verbose=True):
        # shape = self.__input_shape__[1:]

        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.selective_scan": selective_scan_flop_jit,
            "prim::PythonOp.selective_scan_fn": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        #return sum(Gflops.values()) * 1e9
        print(f"params {params} FLOPs {sum(gflops.values())}")
        
    
def ssmamba(model_config):
    model = mamba_SS_model(**model_config)
    return model
#model = mamba_SS_model()