import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from .asf_group_former import GroupedPixelEmbedding
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"

# import selective scan ==============================
try:
    import selective_scan_cuda_oflex
except Exception as e:
    #...
    print(f"WARNING: can not import selective_scan_cuda_oflex.", flush=True)
    print(e, flush=True)

try:
    import selective_scan_cuda_core
except Exception as e:
    #...
    print(f"WARNING: can not import selective_scan_cuda_core.", flush=True)
    print(e, flush=True)

try:
    import selective_scan_cuda
except Exception as e:
    ...
    #print(f"WARNING: can not import selective_scan_cuda.", flush=True)
    #print(e, flush=True)


# fvcore flops =======================================
def flops_selective_scan_fn(B=1, L=256, D=768, N=16, with_D=True, with_Z=False, with_complex=False):
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
    assert not with_complex 
    # https://github.com/state-spaces/mamba/issues/110
    flops = 9 * B * L * D * N
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L    
    return flops

# this is only for selective_scan_ref...
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

    flops += get_flops_einsum([[B, D, L], [D, N]], "bdl,dn->bdln")
    if with_Group:
        flops += get_flops_einsum([[B, D, L], [B, N, L], [B, D, L]], "bdl,bnl,bdl->bdln")
    else:
        flops += get_flops_einsum([[B, D, L], [B, D, N, L], [B, D, L]], "bdl,bdnl,bdl->bdln")
  
    in_for_flops = B * D * N   
    if with_Group:
        in_for_flops += get_flops_einsum([[B, D, N], [B, D, N]], "bdn,bdn->bd")
    else:
        in_for_flops += get_flops_einsum([[B, D, N], [B, N]], "bdn,bn->bd")
    flops += L * in_for_flops 
    if with_D:
        flops += B * D * L
    if with_Z:
        flops += B * D * L  
    return flops


def print_jit_input_names(inputs):
    print("input params: ", end=" ", flush=True)
    try: 
        for i in range(10):
            print(inputs[i].debugName(), end=" ", flush=True)
    except Exception as e:
        pass
    print("", flush=True)


# cross selective scan ===============================
class SelectiveScanMamba(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        # assert nrows in [1, 2, 3, 4], f"{nrows}" # 8+ is too slow to compile
        # assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        # all in float
        # if u.stride(-1) != 1:
        #     u = u.contiguous()
        # if delta.stride(-1) != 1:
        #     delta = delta.contiguous()
        # if D is not None and D.stride(-1) != 1:
        #     D = D.contiguous()
        # if B.stride(-1) != 1:
        #     B = B.contiguous()
        # if C.stride(-1) != 1:
        #     C = C.contiguous()
        # if B.dim() == 3:
        #     B = B.unsqueeze(dim=1)
        #     ctx.squeeze_B = True
        # if C.dim() == 3:
        #     C = C.unsqueeze(dim=1)
        #     ctx.squeeze_C = True
        
        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )
        # dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        # dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanCore(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanOflex(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        out, x, *rest = selective_scan_cuda_oflex.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, 1, oflex)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_oflex.bwd(
            u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
        )
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)


class SelectiveScanFake(torch.autograd.Function):
    # comment all checks if inside cross_selective_scan
    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1, backnrows=1, oflex=True):
        ctx.delta_softplus = delta_softplus
        ctx.backnrows = backnrows
        x = delta
        out = u
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out
    
    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        du, ddelta, dA, dB, dC, dD, ddelta_bias = u * 0, delta * 0, A * 0, B * 0, C * 0, C * 0, (D * 0 if D else None), (delta_bias * 0 if delta_bias else None)
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None, None, None)

# =============
def antidiagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_gather(tensor):
    # 取出矩阵所有反斜向的元素并拼接
    B, C, H, W = tensor.size()
    shift = torch.arange(H, device=tensor.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 使用gather进行索引选择
    return tensor.gather(3, expanded_index).transpose(-1,-2).reshape(B, C, H*W)

def diagonal_scatter(tensor_flat, original_shape):
    # 把斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (shift + torch.arange(W, device=tensor_flat.device)) % W  # 利用广播创建索引矩阵[H, W]
    # 扩展索引以适应B和C维度
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 创建一个空的张量来存储反向散布的结果
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, H, W]，考虑到需要使用transpose将H和W调换
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_根据expanded_index将元素放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

def antidiagonal_scatter(tensor_flat, original_shape):
    # 把反斜向元素拼接起来的一维向量还原为最初的矩阵形式
    B, C, H, W = original_shape
    shift = torch.arange(H, device=tensor_flat.device).unsqueeze(1)  # 创建一个列向量[H, 1]
    index = (torch.arange(W, device=tensor_flat.device) - shift) % W  # 利用广播创建索引矩阵[H, W]
    expanded_index = index.unsqueeze(0).unsqueeze(0).expand(B, C, -1, -1)
    # 初始化一个与原始张量形状相同、元素全为0的张量
    result_tensor = torch.zeros(B, C, H, W, device=tensor_flat.device, dtype=tensor_flat.dtype)
    # 将平铺的张量重新变形为[B, C, W, H]，因为操作是沿最后一个维度收集的，需要调整形状并交换维度
    tensor_reshaped = tensor_flat.reshape(B, C, W, H).transpose(-1, -2)
    # 使用scatter_将元素根据索引放回原位
    result_tensor.scatter_(3, expanded_index, tensor_reshaped)
    return result_tensor

class CrossScan_8x(torch.autograd.Function):
    # ZSJ 这里是把图像按照特定方向展平的地方，改变扫描方向可以在这里修改
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        # xs = x.new_empty((B, 4, C, H * W))
        xs = x.new_empty((B, 8, C, H * W))
        # 添加横向和竖向的扫描
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
    
        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x)
        xs[:, 5] = antidiagonal_gather(x)
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        # 把横向和竖向的反向部分再反向回来，并和原来的横向和竖向相加
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, -1, L)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,C,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,C,H,W))

        y_res = y_rb + y_da
        # return y.view(B, -1, H, W)
        return y_res


class CrossMerge_8x(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        # ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)

        y_rb = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        # 把竖向的部分转成横向，然后再相加,再转回最初是的矩阵形式
        y_rb = y_rb[:, 0] + y_rb[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        y_rb = y_rb.view(B, -1, H, W)

        # 把斜向和反斜向的反向部分再反向回来，并和原来的斜向和反斜向相加
        y_da = ys[:, 4:6] + ys[:, 6:8].flip(dims=[-1]).view(B, 2, D, -1)
        # 把斜向和反斜向的部分都转成原来的最初的矩阵形式，再相加
        y_da = diagonal_scatter(y_da[:, 0], (B,D,H,W)) + antidiagonal_scatter(y_da[:, 1], (B,D,H,W))

        y_res = y_rb + y_da
        return y_res.view(B, D, -1)
        # return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        # xs = x.new_empty((B, 4, C, L))
        xs = x.new_empty((B, 8, C, L))

        # 横向和竖向扫描
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        # xs = xs.view(B, 4, C, H, W)

        # 提供斜向和反斜向的扫描
        xs[:, 4] = diagonal_gather(x.view(B,C,H,W))
        xs[:, 5] = antidiagonal_gather(x.view(B,C,H,W))
        xs[:, 6:8] = torch.flip(xs[:, 4:6], dims=[-1])

        # return xs
        return xs.view(B, 8, C, H, W)

class CrossScan_4x(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)


class CrossMerge_4x(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs

class CrossScan_2x(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        #ys = ys.view(B, 2, -1, L)
        #y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        #ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)

class CrossMerge_2x(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        #y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        xs = xs.view(B, 2, C, H, W)
        return xs

class CrossScan_2x_spe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, L = x.shape
        ctx.shape = (B, C, L)
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, L = ctx.shape
        #ys = ys.view(B, 2, -1, L)
        #y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        #ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        return y

class CrossMerge_2x_spe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, L = ys.shape
        ctx.shape = (L)

        #y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        L = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs

class CrossScan_2x_HW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = torch.flip(xs[:, 0], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        #ys = ys.view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        #ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        #y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)

class CrossMerge_2x_HW(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()

        #y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 2, C, L))
        xs[:, 0] = x
        xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
        xs = xs.view(B, 2, C, H, W)
        return xs

# these are for ablations =============
class CrossScan_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 4, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.flatten(2, 3)
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        L = H * W
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
        return y.view(B, -1, H, W)



class CrossMerge_Ab_2direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        ys = ys.view(B, K, D, -1)
        ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
        y = ys.sum(dim=1)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 4, C, L))
        xs[:, 0] = x
        xs[:, 1] = x
        xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
        xs = xs.view(B, 4, C, H, W)
        return xs


class CrossScan_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        #xs = x.view(B, 1, C, H * W).repeat(1, 4, 1, 1).contiguous()
        xs = x.view(B, 1, C, H * W)
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, H, W = ctx.shape
        #y = ys.sum(dim=1).view(B, C, H, W)
        y = ys.view(B, C, H, W)
        return y


class CrossMerge_Ab_1direction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        #y = ys.sum(dim=1).view(B, D, H * W)
        y = ys.view(B, D, H * W)
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        H, W = ctx.shape
        B, C, L = x.shape
        #xs = x.view(B, 1, C, L).repeat(1, 4, 1, 1).contiguous().view(B, 4, C, H, W)
        xs = x.view(B, 1, C, L).view(B, 1, C, H, W)
        return xs

class CrossScan_1x_spe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, L = x.shape
        ctx.shape = (B, C, L)
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        return xs
    
    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        # out: (b, k, d, l)
        B, C, L = ctx.shape
        #ys = ys.view(B, 2, -1, L)
        #y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        #ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
        y = ys[:, 0]
        return y

class CrossMerge_1x_spe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, L = ys.shape
        ctx.shape = (L)

        #y = ys[:, 0] + ys[:, 1].flip(dims=[-1]).contiguous()
        y = ys[:, 0]
        return y
    
    @staticmethod
    def backward(ctx, x: torch.Tensor):
        # B, D, L = x.shape
        # out: (b, k, d, l)
        L = ctx.shape
        B, C, L = x.shape
        xs = x.new_empty((B, 1, C, L))
        xs[:, 0] = x
        return xs

# =============
# ZSJ 这里是mamba的具体内容，要增加扫描方向就在这里改
def cross_selective_scan(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan_Ab_1direction,
    CrossMerge=CrossMerge_Ab_1direction,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows == 0:
        if D % 4 == 0:
            nrows = 4
        elif D % 3 == 0:
            nrows = 3
        elif D % 2 == 0:
            nrows = 2
        else:
            nrows = 1
        
    if backnrows == 0:
        if D % 4 == 0:
            backnrows = 4
        elif D % 3 == 0:
            backnrows = 3
        elif D % 2 == 0:
            backnrows = 2
        else:
            backnrows = 1

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    xs = CrossScan.apply(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, H, W)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]: # (B, C, H, W)
        y = out_norm(y.view(B, -1, H, W)).permute(0, 2, 3, 1) # (B, H, W, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y).view(B, H, W, -1)

    return (y.to(x.dtype) if to_dtype else y)

def selective_scan_spe(
    x: torch.Tensor=None, 
    x_proj_weight: torch.Tensor=None,
    x_proj_bias: torch.Tensor=None,
    dt_projs_weight: torch.Tensor=None,
    dt_projs_bias: torch.Tensor=None,
    A_logs: torch.Tensor=None,
    Ds: torch.Tensor=None,
    delta_softplus = True,
    out_norm: torch.nn.Module=None,
    out_norm_shape="v0",
    # ==============================
    to_dtype=True, # True: final out to dtype
    force_fp32=False, # True: input fp32
    # ==============================
    nrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    backnrows = -1, # for SelectiveScanNRow; 0: auto; -1: disable;
    ssoflex=True, # True: out fp32 in SSOflex; else, SSOflex is the same as SSCore
    # ==============================
    SelectiveScan=None,
    CrossScan=CrossScan_2x_spe,
    CrossMerge=CrossMerge_2x_spe,
):
    # out_norm: whatever fits (B, L, C); LayerNorm; Sigmoid; Softmax(dim=1);...

    B, D, L = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape

    def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True):
        return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, backnrows, ssoflex)
    
    xs = CrossScan.apply(x)
    
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L)
    dts = dts.contiguous().view(B, -1, L)
    As = -torch.exp(A_logs.to(torch.float)) # (k * c, d_state)
    Bs = Bs.contiguous()
    Cs = Cs.contiguous()
    Ds = Ds.to(torch.float) # (K * c)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    if force_fp32:
        xs = xs.to(torch.float)
        dts = dts.to(torch.float)
        Bs = Bs.to(torch.float)
        Cs = Cs.to(torch.float)
    # ZSJ 这里把矩阵拆分成不同方向的序列，并进行扫描
    ys: torch.Tensor = selective_scan(
        xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus
    ).view(B, K, -1, L)
    # ZSJ 这里把处理之后的序列融合起来，并还原回原来的矩阵形式
    y: torch.Tensor = CrossMerge.apply(ys)

    if out_norm_shape in ["v1"]: # (B, C, L)
        y = out_norm(y.view(B, -1, L)).permute(0, 2, 1) # (B, L, C)
    else: # (B, L, C)
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = out_norm(y)

    return (y.to(x.dtype) if to_dtype else y)

def selective_scan_flop_jit(inputs, outputs):
    print_jit_input_names(inputs)
    B, D, L = inputs[0].type().sizes()
    N = inputs[2].type().sizes()[1]
    flops = flops_selective_scan_fn(B=B, L=L, D=D, N=N, with_D=True, with_Z=False)
    return flops


# =====================================================

class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)

        return x


class OSSM(nn.Module):
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2.0,
        dt_rank="auto",
        act_layer=nn.SiLU,
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        dropout=0.0,
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        initialize="v0",
        # ======================
        forward_type="v2",
        k_group=1,
        # ======================
        **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_inner = int(ssm_ratio * d_model)
        dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_conv = d_conv

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        # softmax | sigmoid | dwconv | norm ===========================
        if forward_type[-len("none"):] == "none":
            forward_type = forward_type[:-len("none")]
            self.out_norm = nn.Identity()
        elif forward_type[-len("dwconv3"):] == "dwconv3":
            forward_type = forward_type[:-len("dwconv3")]
            self.out_norm = nn.Conv2d(d_inner, d_inner, kernel_size=3, padding=1, groups=d_inner, bias=False)
            self.out_norm_shape = "v1"
        elif forward_type[-len("softmax"):] == "softmax":
            forward_type = forward_type[:-len("softmax")]
            self.out_norm = nn.Softmax(dim=1)
        elif forward_type[-len("sigmoid"):] == "sigmoid":
            forward_type = forward_type[:-len("sigmoid")]
            self.out_norm = nn.Sigmoid()
        else:
            self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v0=self.forward_corev0,
            # v2=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanCore),
            v2=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanCore),
            v3=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex), 
            spev3=partial(self.forward_core_spe, force_fp32=False, SelectiveScan=SelectiveScanOflex), 
            v31d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=partial(
                cross_selective_scan, CrossScan=CrossScan_Ab_1direction, CrossMerge=CrossMerge_Ab_1direction,
            )),
            v32d=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=partial(
                cross_selective_scan, CrossScan=CrossScan_Ab_2direction, CrossMerge=CrossMerge_Ab_2direction,
            )),
            # ===============================
            fake=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanFake),
            v1=partial(self.forward_corev2, force_fp32=True, SelectiveScan=SelectiveScanOflex),
            v01=partial(self.forward_corev2, force_fp32=(not self.disable_force32), SelectiveScan=SelectiveScanMamba),
        )
        if forward_type.startswith("debug"):
            from .ss2d_ablations import SS2D_ForwardCoreSpeedAblations, SS2D_ForwardCoreModeAblations, cross_selective_scanv2
            FORWARD_TYPES.update(dict(
                debugforward_core_mambassm_seq=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_seq, self),
                debugforward_core_mambassm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm, self),
                debugforward_core_mambassm_fp16=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fp16, self),
                debugforward_core_mambassm_fusecs=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecs, self),
                debugforward_core_mambassm_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_mambassm_fusecscm, self),
                debugforward_core_sscore_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_sscore_fusecscm, self),
                debugforward_core_sscore_fusecscm_fwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fwdnrow, self),
                debugforward_core_sscore_fusecscm_bwdnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_bwdnrow, self),
                debugforward_core_sscore_fusecscm_fbnrow=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssnrow_fusecscm_fbnrow, self),
                debugforward_core_ssoflex_fusecscm=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm, self),
                debugforward_core_ssoflex_fusecscm_i16o32=partial(SS2D_ForwardCoreSpeedAblations.forward_core_ssoflex_fusecscm_i16o32, self),
                debugscan_sharessm=partial(self.forward_corev2, force_fp32=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scanv2),
            ))
        self.forward_core = FORWARD_TYPES.get(forward_type, None)
        # ZSJ k_group 指的是扫描的方向
        #k_group = k_groups
        #k_group = 2 if forward_type not in ["debugscan_sharessm"] else 1
        #k_group = 4 if forward_type not in ["debugscan_sharessm"] else 1
        #k_group = 8 if forward_type not in ["debugscan_sharessm"] else 1

        # in proj =======================================
        d_proj = d_inner if self.disable_z else (d_inner * 2)
        self.in_proj = nn.Linear(d_model, d_proj, bias=bias, **factory_kwargs)
        self.act: nn.Module = act_layer()
        
        # conv =======================================
        if d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_inner,
                out_channels=d_inner,
                groups=d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(k_group)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K, N, inner)
        del self.x_proj
        
        # out proj =======================================
        self.out_proj = nn.Linear(d_inner, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        if initialize in ["v0"]:
            # dt proj ============================
            self.dt_projs = [
                self.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
                for _ in range(k_group)
            ]
            self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K, inner, rank)
            self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K, inner)
            del self.dt_projs
            
            # A, D =======================================
            self.A_logs = self.A_log_init(d_state, d_inner, copies=k_group, merge=True) # (K * D, N)
            self.Ds = self.D_init(d_inner, copies=k_group, merge=True) # (K * D)
        elif initialize in ["v1"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.randn((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner))) 
        elif initialize in ["v2"]:
            # simple init dt_projs, A_logs, Ds
            self.Ds = nn.Parameter(torch.ones((k_group * d_inner)))
            self.A_logs = nn.Parameter(torch.zeros((k_group * d_inner, d_state))) # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
            self.dt_projs_weight = nn.Parameter(torch.randn((k_group, d_inner, dt_rank)))
            self.dt_projs_bias = nn.Parameter(torch.randn((k_group, d_inner)))
    
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
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

    # only used to run previous version
    def forward_corev0(self, x: torch.Tensor, to_dtype=False, channel_first=False):
        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScanCore.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows, False)

        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        B, D, H, W = x.shape
        D, N = self.A_logs.shape
        K, D, R = self.dt_projs_weight.shape
        L = H * W

        # ZSJ 这里进行data expand操作，也就是把相同的数据在不同方向展开成一维，并拼接起来,但是这个函数只用在旧版本
        # 把横向和竖向拼接在K维度
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        # torch.flip把横向和竖向两个方向都进行反向操作
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float() # (b, k, d_state, l)
        Cs = Cs.float() # (b, k, d_state, l)
        
        As = -torch.exp(self.A_logs.float()) # (k * d, d_state)
        Ds = self.Ds.float() # (k * d)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        # assert len(xs.shape) == 3 and len(dts.shape) == 3 and len(Bs.shape) == 4 and len(Cs.shape) == 4
        # assert len(As.shape) == 2 and len(Ds.shape) == 1 and len(dt_projs_bias.shape) == 1

        out_y = selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
        ).view(B, K, -1, L)
        # assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        y = out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y
        y = y.transpose(dim0=1, dim1=2).contiguous() # (B, L, C)
        y = self.out_norm(y).view(B, H, W, -1)

        return (y.to(x.dtype) if to_dtype else y)
    
    def forward_core_spe(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=selective_scan_spe, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 2, 1).contiguous()

        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x
    
    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanOflex, cross_selective_scan=cross_selective_scan, force_fp32=None):
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        # ZSJ V2版本使用的mamba，要改扫描方向在这里改
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, delta_softplus=True,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            force_fp32=force_fp32,
            SelectiveScan=SelectiveScan,
        )
        return x
    
    def forward(self, x: torch.Tensor, **kwargs):
        with_dconv = (self.d_conv > 1)
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=-1) # (b, h, w, d)
            if not self.disable_z_act:
                z = self.act(z)
        if with_dconv:
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.conv2d(x) # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=with_dconv)
        if not self.disable_z:
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class OSSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        patch_size: int = 7,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        split_ratio=0.5,
        conv_ratio=1.0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm=0,
        spe_branch=0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.ssm_dim = hidden_dim
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.split_ratio = split_ratio > 0
        self.spe_branch = spe_branch > 0
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.spa_dim = patch_size * patch_size
        self.post_norm = post_norm > 0
        try:
            from ss2d_ablations import SS2DDev
            _OSSM = SS2DDev if forward_type.startswith("dev") else OSSM
        except:
            _OSSM = OSSM
        self.dynamic_pos = nn.Conv2d(self.ssm_dim, self.ssm_dim, 3, padding=1, groups=self.ssm_dim)
        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = _OSSM(
                d_model=hidden_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                k_group=1,
            )
        
        if spe_branch:
            self.norm_spe = nn.LayerNorm(self.ssm_dim)
            self.op_spe = _OSSM(
                d_model=1, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type="spe"+forward_type,
                k_group=2,
            )
            self.softmax_spe = nn.Softmax(dim=-1)
            self.softmax_spa = nn.Softmax(dim=-1)
            self.spa_gp = nn.AdaptiveAvgPool2d(1)

        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x_ssm = input
        if self.ssm_branch:  
            x_ssm = x_ssm.permute(0, 3, 1, 2)
            x_ssm = x_ssm + self.dynamic_pos(x_ssm) 
            x_ssm = x_ssm.permute(0, 2, 3, 1)  
            if self.post_norm:
                if self.spe_branch:
                    x_spe_ssm = x_ssm[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(2)

                    attn_spe = self.softmax_spe(self.norm_spe(self.op_spe(x_spe_ssm).permute(0, 2, 1))).unsqueeze(1)
                    x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm) * attn_spe))
                    #x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2).flatten(2, 3).reshape(-1, int(np.sqrt(self.ssm_dim)), int(np.sqrt(self.ssm_dim)), self.spa_dim)
                    #x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm))) + self.drop_path(self.norm_spe(self.op_spe(x_spe_ssm)))
                else:
                    x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm)))
            else:
                if self.spe_branch:
                    x_spe_ssm = self.norm_spe(x_ssm[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(1))
                    x_spe_ssm = self.op_spe(x_spe_ssm.permute(0, 2, 1)).permute(0, 2, 1)
                    attn_spe = self.softmax_spe(x_spe_ssm).unsqueeze(1)
                    x_spa_ssm = self.op(self.norm(x_ssm))
                    attn_spa = self.softmax_spa(self.spa_gp(x_spa_ssm.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
                    #attn_spa = self.softmax_spa(x_spa_ssm[:, self.patch_size // 2, self.patch_size // 2, :]).unsqueeze(1)
                    x_spa_ssm = x_spa_ssm * attn_spe
                    x_spe_ssm = x_spe_ssm * attn_spa
                    x_spa_ssm[:, self.patch_size // 2, self.patch_size // 2, :] += x_spe_ssm[:, 0, :]
                    x_ssm = x_ssm + self.drop_path(x_spa_ssm)
                    #x_spe_ssm = x_ssm.clone()
                    #x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2).flatten(2, 3).reshape(-1, int(np.sqrt(self.ssm_dim)), int(np.sqrt(self.ssm_dim)), self.spa_dim)
                    #x_ssm = x_ssm + self.drop_path(self.op(self.norm(x_ssm))) + self.drop_path(self.op_spe(self.norm_spe(x_spe_ssm))).reshape(-1, self.ssm_dim, self.patch_size, self.patch_size).permute(0, 2, 3, 1)
                else:
                    x_ssm = x_ssm + self.drop_path(self.op(self.norm(x_ssm)))
        #x_fuse = torch.cat([x_ssm, x_conv], dim=-1)
        #x = self.fc2(x_fuse)
        '''
        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN'''
        return x_ssm

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class ConvBranch_3dconv(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 1, (3, 3, 3), padding=(1, 1, 1), bias=False),
            nn.BatchNorm3d(1),
            nn.SiLU(inplace=True)
        )
        '''
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )'''
    
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        
        #x = x + self.conv2(x)
        #x = self.conv3(x)
        x = x.squeeze(1)
        return x

class ConvBranch_HMCB(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, residual = True):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.residual = residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        '''self.conv4 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, bias=False),
            nn.SiLU(inplace=True)
        )'''
    
    def forward(self, x):
        x = self.conv1(x)
        if self.residual:
            x = x + self.conv2(x)
        else:
            x = self.conv2(x)
        #x = x + self.conv2(x)
        x = self.conv3(x)
        '''x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)'''
        return x
    
class ConvBranch_mobilev3(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, residual = True, channel_attn=False):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.residual = residual
        self.channel_attn = channel_attn
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, 3, padding=1, groups=hidden_features, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(2 * hidden_features, hidden_features, 1, bias=False),
            nn.BatchNorm2d(hidden_features),
            nn.SiLU(inplace=True)
        )
        self.conv_spe = nn.Sequential(nn.Conv3d(1, 1, (1, 1, 3), padding=(0, 0, 1), bias=False),
                                     nn.BatchNorm3d(1),
                                     nn.SiLU(inplace=True))
        
        if channel_attn:
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.attn = nn.Sequential(
                nn.Linear(hidden_features, hidden_features//4),
                nn.SiLU(),
                nn.Linear(hidden_features//4, hidden_features),
                nn.Sigmoid()
            )
        #self.fusion = Fusion_adaptive_conv(hidden_features)
        
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out_spa = self.conv2(out)
        out_spe = out.unsqueeze(1)
        out_spe = self.conv_spe(out_spe).squeeze(1)
        out = torch.cat([out_spa, out_spe], dim=1)
        #out = self.fusion(out, out_spe)
        if self.channel_attn:
            x_attn = self.pool(out).squeeze()
            x_attn = self.attn(x_attn).unsqueeze(2).unsqueeze(3)
            out = out * x_attn
        if self.residual:
            out = identity + self.conv3(out)
        else:
            out = self.conv3(out)
        
        '''x = x + self.conv4(x)
        x = self.conv5(x)
        x = x + self.conv6(x)
        x = self.conv7(x)'''
        return out

class Fusion_adaptive(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inter_dim = max(dim, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, inter_dim, 1),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(inter_dim, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=)

    def forward(self, x, y):
        initial = x + y
        feats = initial.permute(0, 3, 1, 2)
        gap = self.pool(feats)
        atten = self.fc2(self.fc1(gap))
        w_conv = self.sigmoid(atten)

        result = initial + (1 - w_conv) * x + w_conv * y
        
        #w_conv = w_conv.squeeze(-1)
        return result

class Fusion_adaptive_conv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inter_dim = max(dim, 32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, inter_dim, 1),
            nn.BatchNorm2d(inter_dim),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(inter_dim, 1, 1),
            nn.BatchNorm2d(1)
        )
        self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax(dim=)

    def forward(self, x, y):
        initial = x + y
        gap = self.pool(initial)
        atten = self.fc2(self.fc1(gap))
        w_conv = self.sigmoid(atten)

        result = (1 - w_conv) * x + w_conv * y

        return result

def channel_norm(x):
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x = (x - mean) / std
    return x

class ASF_SSBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        patch_size: int = 7,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        split_ratio=0.5,
        conv_ratio=1.0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm=0,
        spe_branch=0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.split_ratio = split_ratio > 0
        self.spe_branch = spe_branch > 0
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.spa_dim = patch_size * patch_size
        self.post_norm = post_norm > 0
        if self.split_ratio:
            self.ssm_dim = int(hidden_dim * split_ratio)
            self.conv_dim = hidden_dim - self.ssm_dim
        else:
            self.ssm_dim = hidden_dim
            self.conv_dim = hidden_dim
        
        try:
            from ss2d_ablations import SS2DDev
            _OSSM = SS2DDev if forward_type.startswith("dev") else OSSM
        except:
            _OSSM = OSSM
        self.dynamic_pos = nn.Conv2d(self.ssm_dim, self.ssm_dim, 3, padding=1, groups=self.ssm_dim)

        if self.ssm_branch:
            self.norm = norm_layer(self.ssm_dim)
            self.op = _OSSM(
                d_model=self.ssm_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                k_group=1,
            )
        if spe_branch:
            self.norm_spe = nn.LayerNorm(self.ssm_dim)
            self.op_spe = _OSSM(
                d_model=1, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type="spe"+forward_type,
                k_group=2,
            )
            self.softmax_spe = nn.Softmax(dim=-1)
            self.softmax_spa = nn.Softmax(dim=-1)
            self.spa_gp = nn.AdaptiveAvgPool2d(1)
        conv_hidden_dim = int(self.conv_dim * conv_ratio)
        self.conv_branch = ConvBranch_mobilev3(in_features = self.conv_dim, hidden_features = conv_hidden_dim, out_features=self.ssm_dim)
        #self.conv_branch = ConvBranch_3dconv(in_features = self.conv_dim, hidden_features = conv_hidden_dim, out_features=self.ssm_dim)
        self.fuse = Fusion_adaptive(self.ssm_dim)
        #self.fc2 = nn.Linear(self.ssm_dim, hidden_dim, 1)
        #self.fc2 = nn.Linear(self.ssm_dim + self.conv_dim, hidden_dim, 1)
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

        # visualization
        #self.x_ssm_output = None
        #self.original_ssm_spa_output = None
        #self.attn_spa_output = None
        #self.attn_spe_output = None

    def _forward(self, input: torch.Tensor):
        x_emb = input
        #x_emb = x_emb.permute(0, 3, 1, 2)
        #x_emb = x_emb + self.dynamic_pos(x_emb) 
        #x_emb = x_emb.permute(0, 2, 3, 1)
        if self.split_ratio:
            x_ssm = x_emb[:, :, :, 0:self.ssm_dim]
            x_conv = x_emb[:, :, :, self.ssm_dim:]
        else:
            x_ssm = x_emb.clone()
            x_conv = x_emb.clone()
        if self.ssm_branch:  
            x_ssm = x_ssm.permute(0, 3, 1, 2)
            x_ssm = x_ssm + self.dynamic_pos(x_ssm) 
            x_ssm = x_ssm.permute(0, 2, 3, 1)  
            if self.post_norm:
                if self.spe_branch:
                    x_spe_ssm = x_ssm[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(2)

                    attn_spe = self.softmax_spe(self.norm_spe(self.op_spe(x_spe_ssm).permute(0, 2, 1))).unsqueeze(1)
                    x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm) * attn_spe))
                    #x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2).flatten(2, 3).reshape(-1, int(np.sqrt(self.ssm_dim)), int(np.sqrt(self.ssm_dim)), self.spa_dim)
                    #x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm))) + self.drop_path(self.norm_spe(self.op_spe(x_spe_ssm)))
                else:
                    x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm)))
            else:
                if self.spe_branch:
                    x_spe_ssm = self.norm_spe(x_ssm[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(1))
                    x_spe_ssm = self.op_spe(x_spe_ssm.permute(0, 2, 1)).permute(0, 2, 1)
                    attn_spe = self.softmax_spe(x_spe_ssm).unsqueeze(1)
                    x_spa_ssm = self.op(self.norm(x_ssm))
                    
                    #vis
                    #self.original_ssm_spa_output = x_spa_ssm.detach().clone()

                    attn_spa = self.softmax_spa(self.spa_gp(x_spa_ssm.permute(0, 3, 1, 2)).squeeze(-1).permute(0, 2, 1))
                    #attn_spa = self.softmax_spa(x_spa_ssm[:, self.patch_size // 2, self.patch_size // 2, :]).unsqueeze(1)
                    x_spa_ssm = x_spa_ssm * attn_spe
                    x_spe_ssm = x_spe_ssm * attn_spa
                    x_spa_ssm[:, self.patch_size // 2, self.patch_size // 2, :] += x_spe_ssm[:, 0, :]
                    x_ssm = x_ssm + self.drop_path(x_spa_ssm)
    
                    #x_spe_ssm = x_ssm.clone()
                    #x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2).flatten(2, 3).reshape(-1, int(np.sqrt(self.ssm_dim)), int(np.sqrt(self.ssm_dim)), self.spa_dim)
                    #x_ssm = x_ssm + self.drop_path(self.op(self.norm(x_ssm))) + self.drop_path(self.op_spe(self.norm_spe(x_spe_ssm))).reshape(-1, self.ssm_dim, self.patch_size, self.patch_size).permute(0, 2, 3, 1)
                    
                    #vis
                    #self.x_ssm_output = x_ssm.detach().clone()
                    #self.attn_spa_output = attn_spa.detach().clone()
                    #self.attn_spe_output = attn_spe.detach().clone()

                else:
                    x_ssm = x_ssm + self.drop_path(self.op(self.norm(x_ssm)))
        x_conv = x_conv.permute(0, 3, 1, 2)
        x_conv = self.conv_branch(x_conv)
        x_conv = x_conv.permute(0, 2, 3, 1)
        x = self.fuse(x_ssm, x_conv)
        #x_fuse = torch.cat([x_ssm, x_conv], dim=-1)
        #x = self.fc2(x_fuse)

        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class DynamicFusion(nn.Module):
    def __init__(self, D):
        super(DynamicFusion, self).__init__()
        self.weight_layer = nn.Conv2d(D, D, kernel_size=1, bias=True)  # 1x1 卷积生成融合权重

    def forward(self, x_spe_ssm, x_spa_ssm):
        """
        x_spe_ssm: (B, p, p, D)
        x_spa_ssm: (B, p, p, D)
        return: (B, p, p, D) 融合后的特征
        """
        B, p, p, D = x_spe_ssm.shape  # 获取动态大小

        # 变换到 (B, D, p, p) 以便做通道注意力计算
        x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2)
        x_spa_ssm = x_spa_ssm.permute(0, 3, 1, 2)

        # 计算融合权重 (B, D, p, p)
        alpha = torch.sigmoid(self.weight_layer(x_spe_ssm + x_spa_ssm))  # 动态计算权重
       
        # 融合
        fused = alpha * x_spe_ssm + (1 - alpha) * x_spa_ssm  # 加权求和

        # 变回 (B, p, p, D)
        fused = fused.permute(0, 2, 3, 1)
        return fused
    
class SEMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        # =============================
        patch_size: int = 7,
        ssm_d_state: int = 16,
        ssm_ratio=2.0,
        ssm_dt_rank: Any = "auto",
        ssm_act_layer=nn.SiLU,
        ssm_conv: int = 3,
        ssm_conv_bias=True,
        ssm_drop_rate: float = 0,
        ssm_init="v0",
        forward_type="v2",
        # =============================
        split_ratio=0.5,
        conv_ratio=1.0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
        # =============================
        use_checkpoint: bool = False,
        post_norm=0,
        spe_branch=0,
        **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.split_ratio = split_ratio > 0
        self.spe_branch = spe_branch > 0
        self.use_checkpoint = use_checkpoint
        self.patch_size = patch_size
        self.spa_dim = patch_size * patch_size
        self.post_norm = post_norm > 0
        if self.split_ratio:
            self.ssm_dim = int(hidden_dim * split_ratio)
            self.conv_dim = hidden_dim - self.ssm_dim
        else:
            self.ssm_dim = hidden_dim
            self.conv_dim = hidden_dim
        
        try:
            from ss2d_ablations import SS2DDev
            _OSSM = SS2DDev if forward_type.startswith("dev") else OSSM
        except:
            _OSSM = OSSM
        self.dynamic_pos = nn.Conv2d(self.ssm_dim, self.ssm_dim, 3, padding=1, groups=self.ssm_dim)

        if self.ssm_branch:
            self.norm = norm_layer(self.ssm_dim)
            self.op = _OSSM(
                d_model=self.ssm_dim, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
                k_group=4,
            )
        if spe_branch:
            self.norm_spe = nn.LayerNorm(self.patch_size * self.patch_size)
            self.op_spe = _OSSM(
                d_model=self.patch_size * self.patch_size, 
                d_state=ssm_d_state, 
                ssm_ratio=ssm_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type="spe"+forward_type,
                k_group=2,
            )
            
        #conv_hidden_dim = int(self.conv_dim * conv_ratio)
        #self.conv_branch = ConvBranch_mobilev3(in_features = self.conv_dim, hidden_features = conv_hidden_dim, out_features=self.ssm_dim)
        #self.conv_branch = ConvBranch_3dconv(in_features = self.conv_dim, hidden_features = conv_hidden_dim, out_features=self.ssm_dim)
        #self.fuse = Fusion_adaptive(self.ssm_dim)
        #self.fc2 = nn.Linear(self.ssm_dim, hidden_dim, 1)
        #self.fc2 = nn.Linear(self.ssm_dim + self.conv_dim, hidden_dim, 1)
        self.dyn_fuse = DynamicFusion(self.ssm_dim)
        self.drop_path = DropPath(drop_path)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer, drop=mlp_drop_rate, channels_first=False)

        # visualization
        #self.x_ssm_output = None
        #self.original_ssm_spa_output = None
        #self.attn_spa_output = None
        #self.attn_spe_output = None

    def _forward(self, input: torch.Tensor):
        x_ssm = input
        #x_emb = x_emb.permute(0, 3, 1, 2)
        #x_emb = x_emb + self.dynamic_pos(x_emb) 
        #x_emb = x_emb.permute(0, 2, 3, 1)

        if self.ssm_branch:  
            if self.post_norm:
                if self.spe_branch:
                    x_spe_ssm = x_ssm[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(2)

                    attn_spe = self.softmax_spe(self.norm_spe(self.op_spe(x_spe_ssm).permute(0, 2, 1))).unsqueeze(1)
                    x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm) * attn_spe))
                    #x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2).flatten(2, 3).reshape(-1, int(np.sqrt(self.ssm_dim)), int(np.sqrt(self.ssm_dim)), self.spa_dim)
                    #x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm))) + self.drop_path(self.norm_spe(self.op_spe(x_spe_ssm)))
                else:
                    x_ssm = x_ssm + self.drop_path(self.norm(self.op(x_ssm)))
            else:
                if self.spe_branch:
                    #x_spe_ssm = self.norm_spe(x_ssm[:, self.patch_size // 2, self.patch_size // 2, :].unsqueeze(1))
                    x_spe_ssm = x_ssm.permute(0, 3, 1, 2)
                    x_spe_ssm = self.norm_spe(x_spe_ssm.flatten(2))
                    x_spe_ssm = self.op_spe(x_spe_ssm)
                    x_spe_ssm = x_spe_ssm.view(*x_spe_ssm.shape[:2], self.patch_size, self.patch_size).permute(0, 2, 3, 1)
                    x_spa_ssm = self.op(self.norm(x_ssm))
                    x_fuse = self.dyn_fuse(x_spe_ssm, x_spa_ssm)

                    x_ssm = x_ssm + self.drop_path(x_fuse)
                    
                    #x_spe_ssm = x_ssm.clone()
                    #x_spe_ssm = x_spe_ssm.permute(0, 3, 1, 2).flatten(2, 3).reshape(-1, int(np.sqrt(self.ssm_dim)), int(np.sqrt(self.ssm_dim)), self.spa_dim)
                    #x_ssm = x_ssm + self.drop_path(self.op(self.norm(x_ssm))) + self.drop_path(self.op_spe(self.norm_spe(x_spe_ssm))).reshape(-1, self.ssm_dim, self.patch_size, self.patch_size).permute(0, 2, 3, 1)
                    
                    #vis
                    #self.x_ssm_output = x_ssm.detach().clone()
                    #self.attn_spa_output = attn_spa.detach().clone()
                    #self.attn_spe_output = attn_spe.detach().clone()

                else:
                    x_ssm = x_ssm + self.drop_path(self.op(self.norm(x_ssm)))
        
        #x_fuse = torch.cat([x_ssm, x_conv], dim=-1)
        #x = self.fc2(x_fuse)

        if self.mlp_branch:
            if self.post_norm:
                x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
        return x_ssm

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)

class RSM_SS(nn.Module):
    def __init__(
        self, 
        patch_size=4, 
        in_chans=3, 
        num_classes=1000, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768], 
        # =========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",
        ssm_act_layer="silu",        
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # =========================
        mlp_ratio=4.0,
        mlp_act_layer="gelu",
        mlp_drop_rate=0.0,
        # =========================
        drop_path_rate=0.1, 
        patch_norm=True, 
        norm_layer="LN",
        use_checkpoint=False,  
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.num_features = dims[-1]
        self.dims = dims
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        #_make_patch_embed = self._make_patch_embed_v2
        #self.patch_embed = _make_patch_embed(in_chans, dims[0], patch_size, patch_norm, norm_layer)


        # self.encoder_layers = [nn.ModuleList()] * self.num_layers
        self.encoder_layers = nn.Sequential()
        self.embedding = nn.Linear(in_chans, dims[0])
        self.channel_first = False
        for i_layer in range(self.num_layers):
            self.encoder_layers.append(self._make_layer(
                dim = self.dims[i_layer],
                drop_path = self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                # =================
                patch_size=patch_size,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                # =================
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            ))

        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @staticmethod
    def _make_patch_embed_v2(in_chans=3, embed_dim=96, patch_size=4, patch_norm=True, norm_layer=nn.LayerNorm):
        assert patch_size == 4
        return nn.Sequential(
            nn.Conv2d(in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            (Permute(0, 2, 3, 1) if patch_norm else nn.Identity()),
            (norm_layer(embed_dim // 2) if patch_norm else nn.Identity()),
            (Permute(0, 3, 1, 2) if patch_norm else nn.Identity()),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            Permute(0, 2, 3, 1),
            (norm_layer(embed_dim) if patch_norm else nn.Identity()),
        )

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        patch_size=7,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        # ===========================
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
            ))
        return nn.Sequential(*blocks,)
        
    def forward(self, x: torch.Tensor):
        x = x.squeeze(1)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.embedding(x)
        x = x.reshape(b, h, w, -1)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.classifier(x)

class GroupedPixelEmbedding_mm(nn.Module):
    def __init__(self, in_feature_map_size=7, in_chans=3, embed_dim=128, n_groups=1):
        super().__init__()
        self.ifm_size = in_feature_map_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=3, stride=1, padding=1, groups=n_groups)
        self.batch_norm = nn.BatchNorm2d(embed_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        input: (B, in_feature_map_size, in_feature_map_size, in_chans)
        output: (B, in_feature_map_size, in_feature_map_size, out_chans)
        """
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = self.relu(self.batch_norm(x))
        
        x = x.permute(0, 2, 3, 1)
        
        #after_feature_map_size = self.ifm_size  
        
        return x

class RSM_Group(nn.Module):
    def __init__(
            self, 
            n_groups=[4,4,4,4], 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            ssm_d_state=16, 
            ssm_ratio=2, 
            ssm_dt_rank="auto", 
            ssm_act_layer="silu", 
            ssm_conv=3, 
            ssm_conv_bias=True, 
            ssm_drop_rate=0,
            ssm_init="v0", 
            forward_type="v2",
            spe_branch=0, 
            split_ratio=0, 
            mlp_ratio=4, 
            mlp_act_layer="gelu", 
            mlp_drop_rate=0, 
            drop_path_rate=0.1, 
            patch_norm=True, 
            norm_layer="LN", 
            use_checkpoint=False, 
            post_norm=0,
            **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.encoder_layers = nn.Sequential()
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]
        
        self.group_emb = GroupedPixelEmbedding_mm(
                    in_feature_map_size=patch_size,
                    in_chans=new_bands,
                    embed_dim=dims[0],
                    n_groups=n_groups[0]
                    )
        
        for i_layer in range(self.num_layers):
            self.encoder_layers.append(
                self._make_layer(
                dim = self.dims[i_layer],
                drop_path = self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                patch_size=patch_size,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                post_norm=post_norm,
                # =================
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            )
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        patch_size=7,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        spe_branch=0,
        post_norm=0,
        # ===========================
        split_ratio=0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(OSSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                patch_size=patch_size,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                post_norm=post_norm,
            ))
        return nn.Sequential(*blocks,)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.pad(x).squeeze(1)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.group_emb(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return self.classifier(x)

'''
class ASF_RSM_Group(nn.Module):
    def __init__(
            self, 
            n_groups=[4,4,4,4], 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            ssm_d_state=16, 
            ssm_ratio=2, 
            ssm_dt_rank="auto", 
            ssm_act_layer="silu", 
            ssm_conv=3, 
            ssm_conv_bias=True, 
            ssm_drop_rate=0,
            ssm_init="v0", 
            forward_type="v2",
            spe_branch=0, 
            split_ratio=0, 
            mlp_ratio=4, 
            mlp_act_layer="gelu", 
            mlp_drop_rate=0, 
            drop_path_rate=0.1, 
            patch_norm=True, 
            norm_layer="LN", 
            use_checkpoint=False, 
            post_norm=0,
            **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.encoder_layers = nn.Sequential()
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]
        
        for i_layer in range(self.num_layers):
            self.encoder_layers.append(nn.Sequential(OrderedDict(
                group_emb=GroupedPixelEmbedding_mm(
                    in_feature_map_size=patch_size,
                    in_chans=new_bands if i_layer == 0 else dims[i_layer - 1],
                    embed_dim=dims[i_layer],
                    n_groups=n_groups[i_layer]
                    ),
                ssm_block=self._make_layer(
                dim = self.dims[i_layer],
                drop_path = self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                patch_size=patch_size,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                post_norm=post_norm,
                # =================
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            )))
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        patch_size=7,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        spe_branch=0,
        post_norm=0,
        # ===========================
        split_ratio=0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(ASF_SSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                patch_size=patch_size,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                post_norm=post_norm,
            ))
        return nn.Sequential(*blocks,)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.pad(x).squeeze(1)
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        for layer in self.encoder_layers:
            x = layer(x)
        return self.classifier(x)
'''

class ASF_RSM_Group(nn.Module):
    def __init__(
            self, 
            n_groups=[4,4,4,4], 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            ssm_d_state=16, 
            ssm_ratio=2, 
            ssm_dt_rank="auto", 
            ssm_act_layer="silu", 
            ssm_conv=3, 
            ssm_conv_bias=True, 
            ssm_drop_rate=0,
            ssm_init="v0", 
            forward_type="v2",
            spe_branch=0, 
            split_ratio=0, 
            mlp_ratio=4, 
            mlp_act_layer="gelu", 
            mlp_drop_rate=0, 
            drop_path_rate=0.1, 
            patch_norm=True, 
            norm_layer="LN", 
            use_checkpoint=False, 
            post_norm=0,
            **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.encoder_layers = nn.Sequential()
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        self.group_emb = GroupedPixelEmbedding_mm(
                    in_feature_map_size=patch_size,
                    in_chans=new_bands,
                    embed_dim=dims[0],
                    n_groups=n_groups[0]
                    )
        '''self.group_emb = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, 1, 1, groups=in_chans),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chans, dims[0], 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True))'''
        
        for i_layer in range(self.num_layers):
            self.encoder_layers.append(
                self._make_layer(
                dim = self.dims[i_layer],
                drop_path = self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                patch_size=patch_size,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                post_norm=post_norm,
                # =================
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            )
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

        #vis
        #self.x_ssm_outputs = []
        #self.attn_spa_outputs = []
        #self.attn_spe_outputs = []
        #self.original_ssm_spa_output = []

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        patch_size=7,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        spe_branch=0,
        post_norm=0,
        # ===========================
        split_ratio=0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(ASF_SSBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                patch_size=patch_size,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                post_norm=post_norm,
            ))
        return nn.Sequential(*blocks,)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.pad(x).squeeze(1)
        #x = x.squeeze(1)
        #b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.group_emb(x)
        
        #vis
        #self.x_ssm_outputs.clear()
        #self.attn_spa_outputs.clear()
        
        #self.attn_spe_outputs.clear()
        #self.original_ssm_spa_output.clear()
        for block in self.encoder_layers:
            x = block(x)
        
            #vis
            #if isinstance(block[0], ASF_SSBlock):
            #    if hasattr(block[0], 'x_ssm_output') and block[0].x_ssm_output is not None:
            #        self.x_ssm_outputs.append(block[0].x_ssm_output)
            #    if hasattr(block[0], 'attn_spa_output') and block[0].attn_spa_output is not None:
            #        self.attn_spa_outputs.append(block[0].attn_spa_output)
            #    if hasattr(block[0], 'attn_spe_output') and block[0].attn_spe_output is not None:
            #        self.attn_spe_outputs.append(block[0].attn_spe_output)
            #    if hasattr(block[0], 'original_ssm_spa_output') and block[0].original_ssm_spa_output is not None:
            #        self.original_ssm_spa_output.append(block[0].original_ssm_spa_output)
        
        return self.classifier(x)
    

    def flops(self, shape=(1, 32, 15, 15), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        #return sum(Gflops.values()) * 1e9
        print(f"params {params} GFLOPs {sum(Gflops.values())}")

class SE_Mamba_Group(nn.Module):
    def __init__(
            self, 
            n_groups=[4,4,4,4], 
            patch_size=4, 
            in_chans=3, 
            num_classes=1000, 
            depths=[2, 2, 9, 2], 
            dims=[96, 192, 384, 768], 
            ssm_d_state=16, 
            ssm_ratio=2, 
            ssm_dt_rank="auto", 
            ssm_act_layer="silu", 
            ssm_conv=3, 
            ssm_conv_bias=True, 
            ssm_drop_rate=0,
            ssm_init="v0", 
            forward_type="v2",
            spe_branch=0, 
            split_ratio=0, 
            mlp_ratio=4, 
            mlp_act_layer="gelu", 
            mlp_drop_rate=0, 
            drop_path_rate=0.1, 
            patch_norm=True, 
            norm_layer="LN", 
            use_checkpoint=False, 
            post_norm=0,
            **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.num_features = dims[-1]
        self.dims = dims
        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.encoder_layers = nn.Sequential()
        new_bands = math.ceil(in_chans / n_groups[0]) * n_groups[0]
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, new_bands - in_chans))
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            bn=nn.BatchNorm2d,
        )

        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )

        if isinstance(norm_layer, str) and norm_layer.lower() in ["ln"]:
            norm_layer: nn.Module = _NORMLAYERS[norm_layer.lower()]

        if isinstance(ssm_act_layer, str) and ssm_act_layer.lower() in ["silu", "gelu", "relu"]:
            ssm_act_layer: nn.Module = _ACTLAYERS[ssm_act_layer.lower()]

        if isinstance(mlp_act_layer, str) and mlp_act_layer.lower() in ["silu", "gelu", "relu"]:
            mlp_act_layer: nn.Module = _ACTLAYERS[mlp_act_layer.lower()]

        self.group_emb = GroupedPixelEmbedding_mm(
                    in_feature_map_size=patch_size,
                    in_chans=new_bands,
                    embed_dim=dims[0],
                    n_groups=n_groups[0]
                    )
        '''self.group_emb = nn.Sequential(
            nn.Conv2d(in_chans, in_chans, 3, 1, 1, groups=in_chans),
            nn.BatchNorm2d(in_chans),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_chans, dims[0], 1),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True))'''
        
        for i_layer in range(self.num_layers):
            self.encoder_layers.append(
                self._make_layer(
                dim = self.dims[i_layer],
                drop_path = self.dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=norm_layer,
                patch_size=patch_size,
                # =================
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                post_norm=post_norm,
                # =================
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
            )
            )
        self.classifier = nn.Sequential(OrderedDict(
            norm=norm_layer(self.num_features), # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

        #vis
        #self.x_ssm_outputs = []
        #self.attn_spa_outputs = []
        #self.attn_spe_outputs = []
        #self.original_ssm_spa_output = []

    @staticmethod
    def _make_layer(
        dim=96, 
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=nn.LayerNorm,
        downsample=nn.Identity(),
        patch_size=7,
        # ===========================
        ssm_d_state=16,
        ssm_ratio=2.0,
        ssm_dt_rank="auto",       
        ssm_act_layer=nn.SiLU,
        ssm_conv=3,
        ssm_conv_bias=True,
        ssm_drop_rate=0.0, 
        ssm_init="v0",
        forward_type="v2",
        spe_branch=0,
        post_norm=0,
        # ===========================
        split_ratio=0,
        mlp_ratio=4.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate=0.0,
        **kwargs,
    ):
        depth = len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(SEMBlock(
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                patch_size=patch_size,
                ssm_d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_dt_rank=ssm_dt_rank,
                ssm_act_layer=ssm_act_layer,
                ssm_conv=ssm_conv,
                ssm_conv_bias=ssm_conv_bias,
                ssm_drop_rate=ssm_drop_rate,
                ssm_init=ssm_init,
                forward_type=forward_type,
                spe_branch=spe_branch,
                split_ratio=split_ratio,
                mlp_ratio=mlp_ratio,
                mlp_act_layer=mlp_act_layer,
                mlp_drop_rate=mlp_drop_rate,
                use_checkpoint=use_checkpoint,
                post_norm=post_norm,
            ))
        return nn.Sequential(*blocks,)
    
    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor):
        x = self.pad(x).squeeze(1)
        #x = x.squeeze(1)
        #b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b h w c')
        x = self.group_emb(x)
        
        #vis
        #self.x_ssm_outputs.clear()
        #self.attn_spa_outputs.clear()
        
        #self.attn_spe_outputs.clear()
        #self.original_ssm_spa_output.clear()
        for block in self.encoder_layers:
            x = block(x)
        
            #vis
            #if isinstance(block[0], ASF_SSBlock):
            #    if hasattr(block[0], 'x_ssm_output') and block[0].x_ssm_output is not None:
            #        self.x_ssm_outputs.append(block[0].x_ssm_output)
            #    if hasattr(block[0], 'attn_spa_output') and block[0].attn_spa_output is not None:
            #        self.attn_spa_outputs.append(block[0].attn_spa_output)
            #    if hasattr(block[0], 'attn_spe_output') and block[0].attn_spe_output is not None:
            #        self.attn_spe_outputs.append(block[0].attn_spe_output)
            #    if hasattr(block[0], 'original_ssm_spa_output') and block[0].original_ssm_spa_output is not None:
            #        self.original_ssm_spa_output.append(block[0].original_ssm_spa_output)
        
        return self.classifier(x)
    

    def flops(self, shape=(1, 270, 13, 13), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops={
            "aten::silu": None, # as relu is in _IGNORED_OPS
            "aten::neg": None, # as relu is in _IGNORED_OPS
            "aten::exp": None, # as relu is in _IGNORED_OPS
            "aten::flip": None, # as permute is in _IGNORED_OPS
            # "prim::PythonOp.CrossScan": None,
            # "prim::PythonOp.CrossMerge": None,
            "prim::PythonOp.SelectiveScanMamba": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanOflex": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanCore": selective_scan_flop_jit,
            "prim::PythonOp.SelectiveScanNRow": selective_scan_flop_jit
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)

        del model, input
        #return sum(Gflops.values()) * 1e9
        print(f"params {params} GFLOPs {sum(Gflops.values())}")

def vmamba(model_config, dataset, patch_size):
    model = RSM_SS(**model_config)
    if dataset == 'pu':
        model = RSM_SS(patch_size, 
        in_chans=103, 
        num_classes=9, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768])
    elif dataset == 'whulk':
        model = RSM_SS(patch_size, 
        in_chans=270, 
        num_classes=9, 
        depths=[2, 2, 9, 2], 
        dims=[96, 96, 96, 96])
    elif dataset == 'ip':
        model = RSM_SS(**model_config)
    elif dataset == 'hu2018':
        model = RSM_SS(patch_size, 
        in_chans=48, 
        num_classes=20, 
        depths=[2, 2, 9, 2], 
        dims=[96, 192, 384, 768])
    return model

def rsm_group(model_config):
    model = RSM_Group(**model_config)
    return model

def asf_rsm_group(model_config):
    model = ASF_RSM_Group(**model_config)
    return model

def semamba(model_config):
    model = SE_Mamba_Group(**model_config)
    return model


