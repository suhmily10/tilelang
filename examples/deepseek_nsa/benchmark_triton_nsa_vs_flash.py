#!/usr/bin/env python3
# ruff: noqa

import torch
import time
import argparse
from typing import Optional, Union
from einops import rearrange
import triton
import triton.language as tl
from packaging.version import parse

import fla
if parse(fla.__version__) < parse("0.2.1"):
    from fla.ops.common.utils import prepare_token_indices
else:
    from fla.ops.utils import prepare_token_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

# Try to import Flash Attention
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
    print("Flash Attention available")
except (ImportError, Exception) as e:
    FLASH_AVAILABLE = False
    print(f"Flash Attention not available: {e}")
    print("Will use PyTorch SDPA as baseline instead")


# ============ Triton NSA Forward Kernel ============
@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor),
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['BS', 'BK', 'BV'],
)
@triton.jit
def parallel_nsa_fwd_kernel(q, k, v, o_slc, o_swa, lse_slc, lse_swa, scale, block_indices,
                            block_counts, offsets, token_indices, T, H: tl.constexpr,
                            HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                            S: tl.constexpr, BS: tl.constexpr, WS: tl.constexpr, BK: tl.constexpr,
                            BV: tl.constexpr, USE_OFFSETS: tl.constexpr,
                            USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    p_q = tl.make_block_ptr(q + (bos + i_t) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_o_slc = tl.make_block_ptr(o_slc + (bos + i_t) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse_slc = lse_slc + (bos + i_t) * HQ + i_h * G + tl.arange(0, G)
    b_o_slc = tl.zeros([G, BV], dtype=tl.float32)

    b_m_slc = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc_slc = tl.zeros([G], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (T, V), (H * V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))
            b_s_slc = tl.dot(b_q, b_k_slc)
            b_s_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s_slc, float('-inf'))

            b_m_slc, b_mp_slc = tl.maximum(b_m_slc, tl.max(b_s_slc, 1)), b_m_slc
            b_r_slc = tl.exp(b_mp_slc - b_m_slc)
            b_p_slc = tl.exp(b_s_slc - b_m_slc[:, None])
            b_acc_slc = b_acc_slc * b_r_slc + tl.sum(b_p_slc, 1)
            b_o_slc = b_o_slc * b_r_slc[:, None] + tl.dot(b_p_slc.to(b_q.dtype), b_v_slc)

            b_mp_slc = b_m_slc
    b_o_slc = b_o_slc / b_acc_slc[:, None]
    b_m_slc += tl.log(b_acc_slc)
    tl.store(p_o_slc, b_o_slc.to(p_o_slc.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_lse_slc, b_m_slc.to(p_lse_slc.dtype.element_ty))


def parallel_nsa_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.LongTensor,
    block_counts: Union[torch.LongTensor, int],
    block_size: int,
    window_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    token_indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 256"

    grid = (T, NV, B * H)
    o_slc = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)
    o_swa = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device) if window_size > 0 else None
    lse_slc = torch.empty(B, T, HQ, dtype=torch.float, device=q.device)
    lse_swa = torch.empty(B, T, HQ, dtype=torch.float, device=q.device) if window_size > 0 else None

    parallel_nsa_fwd_kernel[grid](
        q=q, k=k, v=v, o_slc=o_slc, o_swa=o_swa,
        lse_slc=lse_slc, lse_swa=lse_swa, scale=scale,
        block_indices=block_indices, block_counts=block_counts,
        offsets=offsets, token_indices=token_indices,
        T=T, H=H, HQ=HQ, G=G, K=K, V=V, S=S, BS=BS, WS=WS, BK=BK, BV=BV,
    )
    return o_slc, lse_slc, o_swa, lse_swa


# ============ Triton NSA Backward Kernels ============
@triton.heuristics({'USE_OFFSETS': lambda args: args['offsets'] is not None})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dkv(q, k, v, lse_slc, lse_swa, delta_slc, delta_swa, do_slc, do_swa, dk,
                                dv, block_mask, offsets, chunk_indices, scale, T, B: tl.constexpr,
                                H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                                V: tl.constexpr, M: tl.constexpr, BS: tl.constexpr,
                                WS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                                USE_OFFSETS: tl.constexpr):
    i_v, i_s, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_s = tl.load(chunk_indices + i_s * 2).to(tl.int32), tl.load(chunk_indices + i_s * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    p_k = tl.make_block_ptr(k + (bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_v = tl.make_block_ptr(v + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))
    p_dk = tl.make_block_ptr(dk + (i_v * B * T * H + bos * H + i_h) * K, (T, K), (H * K, 1), (i_s * BS, 0), (BS, BK), (1, 0))
    p_dv = tl.make_block_ptr(dv + (bos * H + i_h) * V, (T, V), (H * V, 1), (i_s * BS, i_v * BV), (BS, BV), (1, 0))

    b_k = tl.load(p_k, boundary_check=(0, 1))
    b_dk = tl.zeros([BS, BK], dtype=tl.float32)
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_dv = tl.zeros([BS, BV], dtype=tl.float32)

    for i in range(i_s * BS, T):
        b_m_slc = tl.load(block_mask + (bos + i) * H * M + i_h * M + i_s)
        if b_m_slc:
            p_q = tl.make_block_ptr(q + (bos + i) * HQ * K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
            b_q = tl.load(p_q, boundary_check=(0, 1))
            b_q = (b_q * scale).to(b_q.dtype)

            p_do_slc = tl.make_block_ptr(do_slc + (bos + i) * HQ * V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
            p_lse_slc = lse_slc + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            p_delta_slc = delta_slc + (bos + i) * HQ + i_h * G + tl.arange(0, G)
            b_do_slc = tl.load(p_do_slc, boundary_check=(0, 1))
            b_lse_slc = tl.load(p_lse_slc)
            b_delta_slc = tl.load(p_delta_slc)
            b_s_slc = tl.dot(b_k, tl.trans(b_q))
            b_p_slc = tl.exp(b_s_slc - b_lse_slc[None, :])
            b_p_slc = tl.where((i >= (i_s * BS + tl.arange(0, BS)))[:, None], b_p_slc, 0)
            b_dv += tl.dot(b_p_slc.to(b_do_slc.dtype), b_do_slc)
            b_dp_slc = tl.dot(b_v, tl.trans(b_do_slc))
            b_ds_slc = b_p_slc * (b_dp_slc - b_delta_slc[None, :])
            b_dk += tl.dot(b_ds_slc.to(b_q.dtype), b_q)

    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_dv, b_dv.to(p_dv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None,
    'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)
})
@triton.autotune(
    configs=[triton.Config({}, num_warps=num_warps) for num_warps in [1, 2, 4, 8]],
    key=['BS', 'BK', 'BV'],
)
@triton.jit(do_not_specialize=['T'])
def parallel_nsa_bwd_kernel_dq(q, k, v, lse_slc, delta_slc, do_slc, lse_swa, delta_swa, do_swa, dq,
                               scale, block_indices, block_counts, offsets, token_indices, T,
                               B: tl.constexpr, H: tl.constexpr, HQ: tl.constexpr, G: tl.constexpr,
                               K: tl.constexpr, V: tl.constexpr, S: tl.constexpr, BS: tl.constexpr,
                               WS: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                               USE_OFFSETS: tl.constexpr, USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_v, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(token_indices + i_t * 2).to(tl.int32), tl.load(token_indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    q += (bos + i_t) * HQ * K
    do_slc += (bos + i_t) * HQ * V
    lse_slc += (bos + i_t) * HQ
    delta_slc += (bos + i_t) * HQ
    dq += (i_v * B * T + bos + i_t) * HQ * K
    block_indices += (bos + i_t) * H * S + i_h * S

    if USE_BLOCK_COUNTS:
        NS = tl.load(block_counts + (bos + i_t) * H + i_h)
    else:
        NS = S

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V

    p_q = tl.make_block_ptr(q, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))
    p_dq = tl.make_block_ptr(dq, (HQ, K), (K, 1), (i_h * G, 0), (G, BK), (1, 0))

    b_q = tl.load(p_q, boundary_check=(0, 1))
    b_q = (b_q * scale).to(b_q.dtype)

    p_do_slc = tl.make_block_ptr(do_slc, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))
    p_lse_slc = lse_slc + i_h * G + tl.arange(0, G)
    p_delta_slc = delta_slc + i_h * G + tl.arange(0, G)

    b_do_slc = tl.load(p_do_slc, boundary_check=(0, 1))
    b_lse_slc = tl.load(p_lse_slc)
    b_delta_slc = tl.load(p_delta_slc)

    b_dq_slc = tl.zeros([G, BK], dtype=tl.float32)
    for i in range(NS):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t and i_s >= 0:
            p_k_slc = tl.make_block_ptr(k, (K, T), (1, H * K), (0, i_s), (BK, BS), (0, 1))
            p_v_slc = tl.make_block_ptr(v, (V, T), (1, H * V), (i_v * BV, i_s), (BV, BS), (0, 1))
            b_k_slc = tl.load(p_k_slc, boundary_check=(0, 1))
            b_v_slc = tl.load(p_v_slc, boundary_check=(0, 1))

            b_s_slc = tl.dot(b_q, b_k_slc)
            b_p_slc = tl.exp(b_s_slc - b_lse_slc[:, None])
            b_p_slc = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_p_slc, 0)

            b_dp_slc = tl.dot(b_do_slc, b_v_slc)
            b_ds_slc = b_p_slc * (b_dp_slc.to(tl.float32) - b_delta_slc[:, None])
            b_dq_slc += tl.dot(b_ds_slc.to(b_k_slc.dtype), tl.trans(b_k_slc))
    b_dq_slc *= scale

    tl.store(p_dq, b_dq_slc.to(p_dq.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics({'USE_BLOCK_COUNTS': lambda args: isinstance(args['block_counts'], torch.Tensor)})
@triton.jit
def parallel_nsa_kernel_mask(block_indices, block_counts, block_mask, T: tl.constexpr,
                             H: tl.constexpr, S: tl.constexpr, BS: tl.constexpr, NS: tl.constexpr,
                             USE_BLOCK_COUNTS: tl.constexpr):
    i_t, i_b, i_hs = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h, i_s = i_hs // S, i_hs % S

    b_i = tl.load(block_indices + i_b * T * H * S + i_t * H * S + i_h * S + i_s)
    if USE_BLOCK_COUNTS:
        b_m = b_i * BS <= i_t and i_s < tl.load(block_counts + i_b * T * H + i_t * H + i_h)
    else:
        b_m = b_i * BS <= i_t

    if b_i < NS and b_i >= 0:
        tl.store(block_mask + i_b * T * H * NS + i_t * H * NS + i_h * NS + b_i, b_m.to(block_mask.dtype.element_ty))


@triton.jit
def parallel_nsa_bwd_kernel_preprocess(o, do, delta, B: tl.constexpr, V: tl.constexpr):
    i_n = tl.program_id(0)
    o_d = tl.arange(0, B)
    m_d = o_d < V

    b_o = tl.load(o + i_n * V + o_d, mask=m_d, other=0)
    b_do = tl.load(do + i_n * V + o_d, mask=m_d, other=0).to(tl.float32)
    b_delta = tl.sum(b_o * b_do)

    tl.store(delta + i_n, b_delta.to(delta.dtype.element_ty))


def parallel_nsa_block_mask(block_indices: torch.LongTensor, block_counts: Union[torch.LongTensor, int],
                            offsets: torch.LongTensor, block_size: int):
    B, T, H, S = block_indices.shape
    BS = block_size
    if offsets is not None:
        raise NotImplementedError("Variable length not supported in this benchmark")
    else:
        NS = triton.cdiv(T, BS)
    block_mask = torch.zeros(B, T, H, NS, dtype=torch.bool, device=block_indices.device)

    parallel_nsa_kernel_mask[(T, B, H * S)](
        block_indices=block_indices, block_counts=block_counts, block_mask=block_mask,
        T=T, H=H, S=S, BS=BS, NS=NS)
    return block_mask


def parallel_nsa_bwd_preprocess(o: torch.Tensor, do: torch.Tensor):
    V = o.shape[-1]
    delta = torch.empty_like(o[..., 0], dtype=torch.float32)
    parallel_nsa_bwd_kernel_preprocess[(delta.numel(),)](
        o=o, do=do, delta=delta, B=triton.next_power_of_2(V), V=V,)
    return delta


def parallel_nsa_bwd(q, k, v, o_slc, lse_slc, do_slc, o_swa, lse_swa, do_swa,
                     block_indices, block_counts, block_size, window_size, scale,
                     offsets=None, token_indices=None):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    WS = window_size
    BK = triton.next_power_of_2(K)
    BV = min(128, triton.next_power_of_2(v.shape[-1]))
    NV = triton.cdiv(V, BV)

    delta_slc = parallel_nsa_bwd_preprocess(o_slc, do_slc)
    delta_swa = parallel_nsa_bwd_preprocess(o_swa, do_swa) if window_size > 0 else None

    dq = torch.empty(NV, *q.shape, dtype=q.dtype if NV == 1 else torch.float, device=q.device)
    grid = (T, NV, B * H)
    parallel_nsa_bwd_kernel_dq[grid](
        q=q, k=k, v=v, lse_slc=lse_slc, delta_slc=delta_slc, do_slc=do_slc,
        lse_swa=lse_swa, delta_swa=delta_swa, do_swa=do_swa, dq=dq,
        block_indices=block_indices, block_counts=block_counts,
        offsets=offsets, token_indices=token_indices, scale=scale,
        T=T, B=B, H=H, HQ=HQ, G=G, K=K, V=V, S=S, BS=BS, WS=WS, BK=BK, BV=BV)
    dq = dq.sum(0)

    if offsets is not None:
        raise NotImplementedError("Variable length not supported in this benchmark")
    else:
        chunk_indices = None
        NS = triton.cdiv(T, BS)

    block_mask = parallel_nsa_block_mask(block_indices, block_counts, offsets, block_size)
    dk = torch.empty(NV, *k.shape, dtype=k.dtype if NV == 1 else torch.float, device=q.device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=q.device)

    grid = (NV, NS, B * H)
    parallel_nsa_bwd_kernel_dkv[grid](
        q=q, k=k, v=v, lse_slc=lse_slc, lse_swa=lse_swa,
        delta_slc=delta_slc, delta_swa=delta_swa, do_slc=do_slc, do_swa=do_swa,
        dk=dk, dv=dv, block_mask=block_mask, offsets=offsets,
        chunk_indices=chunk_indices, scale=scale,
        T=T, B=B, H=H, HQ=HQ, G=G, K=K, V=V, M=block_mask.shape[-1],
        BS=BS, WS=WS, BK=BK, BV=BV)
    dk = dk.sum(0)
    return dq, dk, dv


class ParallelNSAFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, block_indices, block_counts, block_size, window_size, scale, offsets):
        ctx.dtype = q.dtype
        token_indices = prepare_token_indices(offsets) if offsets is not None else None

        o_slc, lse_slc, o_swa, lse_swa = parallel_nsa_fwd(
            q=q, k=k, v=v, block_indices=block_indices, block_counts=block_counts,
            block_size=block_size, window_size=window_size, scale=scale,
            offsets=offsets, token_indices=token_indices)
        ctx.save_for_backward(q, k, v, o_slc, lse_slc, o_swa, lse_swa)
        ctx.block_indices = block_indices
        ctx.block_counts = block_counts
        ctx.offsets = offsets
        ctx.token_indices = token_indices
        ctx.block_size = block_size
        ctx.window_size = window_size
        ctx.scale = scale
        return o_slc.to(q.dtype), o_swa.to(q.dtype) if o_swa is not None else o_swa

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do_slc, do_swa):
        q, k, v, o_slc, lse_slc, o_swa, lse_swa = ctx.saved_tensors
        dq, dk, dv = parallel_nsa_bwd(
            q=q, k=k, v=v, o_slc=o_slc, o_swa=o_swa,
            lse_slc=lse_slc, lse_swa=lse_swa, do_slc=do_slc, do_swa=do_swa,
            block_indices=ctx.block_indices, block_counts=ctx.block_counts,
            block_size=ctx.block_size, window_size=ctx.window_size, scale=ctx.scale,
            offsets=ctx.offsets, token_indices=ctx.token_indices)
        return dq.to(q), dk.to(k), dv.to(v), None, None, None, None, None, None, None, None


def parallel_nsa(q, k, v, g_slc, g_swa, block_indices, block_counts=None, block_size=64,
                 window_size=0, scale=None, cu_seqlens=None, head_first=False):
    if scale is None:
        scale = k.shape[-1]**-0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
    if head_first:
        q, k, v, block_indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, block_indices))
        g_slc, g_swa = map(lambda x: rearrange(x, 'b h t -> b t h'), (g_slc, g_swa))
        if isinstance(block_counts, torch.Tensor):
            block_counts = rearrange(block_counts, 'b h t -> b t h')
    assert q.shape[2] % (k.shape[2] * 16) == 0, "Group size must be a multiple of 16 in NSA"

    if isinstance(block_counts, int):
        block_indices = block_indices[:, :, :, :block_counts]
        block_counts = None

    o_slc, o_swa = ParallelNSAFunction.apply(q, k, v, block_indices, block_counts, block_size,
                                             window_size, scale, cu_seqlens)
    if window_size > 0:
        o = torch.addcmul(o_slc * g_slc.unsqueeze(-1), o_swa, g_swa.unsqueeze(-1))
    else:
        o = o_slc * g_slc.unsqueeze(-1)
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o


# ============ Benchmark Functions ============
def generate_block_indices(batch, seq_len, heads, selected_blocks, block_size):
    """Generate random block indices for the benchmark."""
    block_indices = torch.full((batch, seq_len, heads, selected_blocks),
                               seq_len, dtype=torch.long, device='cuda')

    for b in range(batch):
        for t in range(seq_len):
            for h in range(heads):
                i_i = torch.randperm(max(1, (t // block_size)))[:selected_blocks]
                block_indices[b, t, h, :len(i_i)] = i_i

    block_indices = block_indices.sort(-1)[0]

    breakpoint()
    return block_indices


def benchmark_triton_nsa(batch_size, seq_len, heads, head_query, dim, selected_blocks, block_size,
                        dtype, scale, warmup=10, iterations=100, test_backward=True):
    """Benchmark the Triton NSA implementation."""
    torch.random.manual_seed(0)

    # Create input tensors
    Q = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype, device='cuda', requires_grad=test_backward)
    K = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device='cuda', requires_grad=test_backward)
    V = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device='cuda', requires_grad=test_backward)
    g_slc = torch.ones((batch_size, seq_len, head_query), dtype=dtype, device='cuda')
    g_swa = torch.ones((batch_size, seq_len, head_query), dtype=dtype, device='cuda')

    # Generate block indices
    block_indices = generate_block_indices(batch_size, seq_len, heads, selected_blocks, block_size)
    block_counts = torch.randint(1, selected_blocks + 1, (batch_size, seq_len, heads), device='cuda')

    # Gradient output for backward pass
    if test_backward:
        grad_output = torch.randn((batch_size, seq_len, head_query, dim), dtype=dtype, device='cuda')

    # Warmup
    for _ in range(warmup):
        output = parallel_nsa(q=Q, k=K, v=V, g_slc=g_slc, g_swa=g_swa,
                             block_indices=block_indices, block_counts=block_counts,
                             block_size=block_size, scale=scale)
        if test_backward:
            output.backward(grad_output, retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None

    # Benchmark Forward Only
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        output = parallel_nsa(q=Q, k=K, v=V, g_slc=g_slc, g_swa=g_swa,
                             block_indices=block_indices, block_counts=block_counts,
                             block_size=block_size, scale=scale)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start_time) / iterations * 1000

    # Benchmark Backward Only (if enabled)
    bwd_time = 0
    if test_backward:
        # Do one forward to get output
        output = parallel_nsa(q=Q, k=K, v=V, g_slc=g_slc, g_swa=g_swa,
                             block_indices=block_indices, block_counts=block_counts,
                             block_size=block_size, scale=scale)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            output.backward(grad_output, retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None
            # Need to do forward again for next backward
            if _ < iterations - 1:
                output = parallel_nsa(q=Q, k=K, v=V, g_slc=g_slc, g_swa=g_swa,
                                     block_indices=block_indices, block_counts=block_counts,
                                     block_size=block_size, scale=scale)
        torch.cuda.synchronize()
        bwd_time = (time.time() - start_time) / iterations * 1000

    # Benchmark Forward + Backward
    total_time = 0
    if test_backward:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            output = parallel_nsa(q=Q, k=K, v=V, g_slc=g_slc, g_swa=g_swa,
                                 block_indices=block_indices, block_counts=block_counts,
                                 block_size=block_size, scale=scale)
            output.backward(grad_output, retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None
        torch.cuda.synchronize()
        total_time = (time.time() - start_time) / iterations * 1000
    else:
        total_time = fwd_time

    # Calculate FLOPs
    flops_per_token = 4 * dim * selected_blocks * block_size
    total_flops = batch_size * seq_len * head_query * flops_per_token
    
    fwd_tflops = total_flops / (fwd_time / 1000) / 1e12
    bwd_tflops = (total_flops * 2) / (bwd_time / 1000) / 1e12 if test_backward else 0
    total_tflops = (total_flops * 3) / (total_time / 1000) / 1e12 if test_backward else fwd_tflops

    return {
        "fwd_time_ms": fwd_time,
        "bwd_time_ms": bwd_time,
        "total_time_ms": total_time,
        "fwd_tflops": fwd_tflops,
        "bwd_tflops": bwd_tflops,
        "total_tflops": total_tflops,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "head_query": head_query,
        "dim": dim,
        "selected_blocks": selected_blocks,
        "block_size": block_size,
        "test_backward": test_backward
    }


def benchmark_flash_attention(batch_size, seq_len, heads, head_query, dim, selected_blocks, block_size,
                              dtype, scale, warmup=10, iterations=100, test_backward=True):
    """Benchmark Flash Attention implementation."""
    if not FLASH_AVAILABLE:
        return None

    torch.random.manual_seed(0)

    # Create input tensors - Flash Attention uses same number of heads for Q, K, V
    Q = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device='cuda', requires_grad=test_backward)
    K = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device='cuda', requires_grad=test_backward)
    V = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device='cuda', requires_grad=test_backward)

    # Gradient output for backward pass
    if test_backward:
        grad_output = torch.randn((batch_size, seq_len, heads, dim), dtype=dtype, device='cuda')

    # Warmup
    for _ in range(warmup):
        output = flash_attn_func(Q, K, V, causal=True, softmax_scale=scale)
        if test_backward:
            output.backward(grad_output, retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None

    # Benchmark Forward Only
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(iterations):
        output = flash_attn_func(Q, K, V, causal=True, softmax_scale=scale)
    torch.cuda.synchronize()
    fwd_time = (time.time() - start_time) / iterations * 1000

    # Benchmark Backward Only (if enabled)
    bwd_time = 0
    if test_backward:
        # Do one forward to get output
        output = flash_attn_func(Q, K, V, causal=True, softmax_scale=scale)
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            output.backward(grad_output, retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None
            # Need to do forward again for next backward
            if _ < iterations - 1:
                output = flash_attn_func(Q, K, V, causal=True, softmax_scale=scale)
        torch.cuda.synchronize()
        bwd_time = (time.time() - start_time) / iterations * 1000

    # Benchmark Forward + Backward
    total_time = 0
    if test_backward:
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(iterations):
            output = flash_attn_func(Q, K, V, causal=True, softmax_scale=scale)
            output.backward(grad_output, retain_graph=True)
            Q.grad = None
            K.grad = None
            V.grad = None
        torch.cuda.synchronize()
        total_time = (time.time() - start_time) / iterations * 1000
    else:
        total_time = fwd_time

    # Calculate FLOPs for full attention
    flops_per_token = 4 * dim * seq_len
    total_flops = batch_size * seq_len * heads * flops_per_token
    
    fwd_tflops = total_flops / (fwd_time / 1000) / 1e12
    bwd_tflops = (total_flops * 2) / (bwd_time / 1000) / 1e12 if test_backward else 0
    total_tflops = (total_flops * 3) / (total_time / 1000) / 1e12 if test_backward else fwd_tflops

    return {
        "fwd_time_ms": fwd_time,
        "bwd_time_ms": bwd_time,
        "total_time_ms": total_time,
        "fwd_tflops": fwd_tflops,
        "bwd_tflops": bwd_tflops,
        "total_tflops": total_tflops,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "heads": heads,
        "head_query": heads,
        "dim": dim,
        "selected_blocks": "N/A",
        "block_size": "N/A",
        "test_backward": test_backward
    }


def run_benchmark_suite(test_backward=True, warmup=5, iterations=50):
    """Run a suite of benchmarks with different configurations."""

    configs = [
        # Default config: Long sequence
        {
            "batch_size": 2,
            "seq_len": 32768,
            "heads": 2,
            "head_query": 32,  # 2 * 16
            "dim": 128,
            "selected_blocks": 32,
            "block_size": 64
        },
    ]

    results = []
    for i, config in enumerate(configs):
        print(f"\n{'='*60}")
        print(f"Configuration {i+1}: {config}")
        print(f"{'='*60}")

        # Benchmark Triton NSA
        print("Benchmarking Triton NSA:")
        triton_result = benchmark_triton_nsa(
            batch_size=config["batch_size"], seq_len=config["seq_len"],
            heads=config["heads"], head_query=config["head_query"], dim=config["dim"],
            selected_blocks=config["selected_blocks"], block_size=config["block_size"],
            dtype=torch.float16, scale=0.1, warmup=warmup, iterations=iterations,
            test_backward=test_backward)
        results.append({"impl": "triton_nsa", **triton_result})
        
        print(f"  Forward:  {triton_result['fwd_time_ms']:.2f} ms, {triton_result['fwd_tflops']:.2f} TFLOPs")
        if test_backward:
            print(f"  Backward: {triton_result['bwd_time_ms']:.2f} ms, {triton_result['bwd_tflops']:.2f} TFLOPs")
            print(f"  Total:    {triton_result['total_time_ms']:.2f} ms, {triton_result['total_tflops']:.2f} TFLOPs")

        # Benchmark Flash Attention
        if FLASH_AVAILABLE:
            print("\nBenchmarking Flash Attention:")
            flash_result = benchmark_flash_attention(
                batch_size=config["batch_size"], seq_len=config["seq_len"],
                heads=config["head_query"], head_query=config["head_query"], dim=config["dim"],
                selected_blocks=config["selected_blocks"], block_size=config["block_size"],
                dtype=torch.float16, scale=0.1, warmup=warmup, iterations=iterations,
                test_backward=test_backward)
            if flash_result:
                results.append({"impl": "flash_attention", **flash_result})
                print(f"  Forward:  {flash_result['fwd_time_ms']:.2f} ms, {flash_result['fwd_tflops']:.2f} TFLOPs")
                if test_backward:
                    print(f"  Backward: {flash_result['bwd_time_ms']:.2f} ms, {flash_result['bwd_tflops']:.2f} TFLOPs")
                    print(f"  Total:    {flash_result['total_time_ms']:.2f} ms, {flash_result['total_tflops']:.2f} TFLOPs")
                
                print("\nComparison (Triton NSA vs Flash Attention):")
                fwd_speedup = flash_result["fwd_time_ms"] / triton_result["fwd_time_ms"]
                print(f"  Forward speedup:  {fwd_speedup:.2f}x")
                if test_backward:
                    bwd_speedup = flash_result["bwd_time_ms"] / triton_result["bwd_time_ms"]
                    total_speedup = flash_result["total_time_ms"] / triton_result["total_time_ms"]
                    print(f"  Backward speedup: {bwd_speedup:.2f}x")
                    print(f"  Total speedup:    {total_speedup:.2f}x")
        else:
            print("\nFlash Attention not available, skipping...")

        print("-" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Triton NSA vs Flash Attention")
    parser.add_argument("--batch", type=int, default=2, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head_query", type=int, default=128, help="Number of query heads")
    parser.add_argument("--dim", type=int, default=64, help="Head dimension")
    parser.add_argument("--selected_blocks", type=int, default=16, help="Number of selected blocks")
    parser.add_argument("--block_size", type=int, default=32, help="Block size")
    parser.add_argument("--dtype", type=str, default="float16", help="Data type (float16 or float32)")
    parser.add_argument("--scale", type=float, default=0.1, help="Attention scale factor")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--suite", action="store_true", help="Run benchmark suite")
    parser.add_argument("--no-backward", action="store_true", help="Skip backward pass testing")

    args = parser.parse_args()

    # For NSA, ensure head_query is a multiple of heads*16
    if args.head_query % (args.heads * 16) != 0:
        args.head_query = ((args.head_query // (args.heads * 16)) + 1) * (args.heads * 16)
        print(f"Adjusted head_query to {args.head_query} to be compatible with NSA implementation")

    test_backward = not args.no_backward

    if args.suite:
        run_benchmark_suite(test_backward=test_backward)
    else:
        dtype = torch.float16 if args.dtype == "float16" else torch.float32

        print("Benchmarking Triton NSA:")
        triton_result = benchmark_triton_nsa(
            batch_size=args.batch, seq_len=args.seq_len, heads=args.heads, head_query=args.head_query,
            dim=args.dim, selected_blocks=args.selected_blocks, block_size=args.block_size,
            dtype=dtype, scale=args.scale, warmup=args.warmup, iterations=args.iterations,
            test_backward=test_backward)
        
        print("\nBenchmark Results (Triton NSA):")
        print(f"Configuration: batch={args.batch}, seq_len={args.seq_len}, heads={args.heads}, " +
              f"head_query={args.head_query}, dim={args.dim}, blocks={args.selected_blocks}, " +
              f"block_size={args.block_size}")
        print(f"  Forward:  {triton_result['fwd_time_ms']:.2f} ms, {triton_result['fwd_tflops']:.2f} TFLOPs")
        if test_backward:
            print(f"  Backward: {triton_result['bwd_time_ms']:.2f} ms, {triton_result['bwd_tflops']:.2f} TFLOPs")
            print(f"  Total:    {triton_result['total_time_ms']:.2f} ms, {triton_result['total_tflops']:.2f} TFLOPs")

        if FLASH_AVAILABLE:
            print("\nBenchmarking Flash Attention:")
            flash_result = benchmark_flash_attention(
                batch_size=args.batch, seq_len=args.seq_len, heads=args.head_query,
                head_query=args.head_query, dim=args.dim, selected_blocks=args.selected_blocks,
                block_size=args.block_size, dtype=dtype, scale=args.scale,
                warmup=args.warmup, iterations=args.iterations, test_backward=test_backward)
            
            if flash_result:
                print("\nBenchmark Results (Flash Attention):")
                print(f"Configuration: batch={args.batch}, seq_len={args.seq_len}, heads={args.head_query}, " +
                      f"dim={args.dim}")
                print(f"  Forward:  {flash_result['fwd_time_ms']:.2f} ms, {flash_result['fwd_tflops']:.2f} TFLOPs")
                if test_backward:
                    print(f"  Backward: {flash_result['bwd_time_ms']:.2f} ms, {flash_result['bwd_tflops']:.2f} TFLOPs")
                    print(f"  Total:    {flash_result['total_time_ms']:.2f} ms, {flash_result['total_tflops']:.2f} TFLOPs")
                
                print("\nComparison (Triton NSA vs Flash Attention):")
                fwd_speedup = flash_result["fwd_time_ms"] / triton_result["fwd_time_ms"]
                print(f"  Forward speedup:  {fwd_speedup:.2f}x")
                if test_backward:
                    bwd_speedup = flash_result["bwd_time_ms"] / triton_result["bwd_time_ms"]
                    total_speedup = flash_result["total_time_ms"] / triton_result["total_time_ms"]
                    print(f"  Backward speedup: {bwd_speedup:.2f}x")
                    print(f"  Total speedup:    {total_speedup:.2f}x")
        else:
            print("\nFlash Attention not available, skipping comparison...")

