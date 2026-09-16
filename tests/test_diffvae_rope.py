"""Interleaved axial RoPE of the diffusion decoder vs a numpy reference."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from ltx_core_mlx.model.video_vae.diffusion_decoder.rope import apply_axial_rope, inv_freqs, rope_dim_split


def test_dim_split_64_is_16_24_24():
    assert rope_dim_split(64) == (16, 24, 24)
    assert rope_dim_split(4) == (2, 1, 1)


def test_inv_freqs_formula_fp32_from_fp64():
    d = 16
    got = np.array(inv_freqs(d))
    ref = (1.0 / 10000.0 ** (np.arange(0, d, 2, dtype=np.float64) / d)).astype(np.float32)
    assert got.dtype == np.float32 and np.array_equal(got, ref)


def _rot_ref(x, pos, inv):
    # x (..., N, d) numpy fp32, interleaved pairs; pos (N,), inv (d/2,)
    # Handle odd dimensions by padding
    d = x.shape[-1]
    is_odd = d % 2 == 1
    if is_odd:
        x = np.pad(x, ((0, 0),) * (x.ndim - 1) + ((0, 1),), mode="constant")

    xe, xo = x[..., 0::2], x[..., 1::2]
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    re, ro = xe * cos - xo * sin, xe * sin + xo * cos
    out = np.empty_like(x)
    out[..., 0::2] = re
    out[..., 1::2] = ro

    if is_odd:
        out = out[..., :d]
    return out


def test_axial_rope_matches_reference_on_each_axis_block():
    B, T, H, W, NH, D = 1, 3, 4, 5, 2, 8
    split = rope_dim_split(D)  # (2, 3, 3)
    inv = tuple(inv_freqs(d) for d in split)
    x = mx.random.normal((B, T, H, W, NH, D))
    t_pos, h_pos, w_pos = (
        mx.arange(T).astype(mx.float32),
        mx.arange(H).astype(mx.float32),
        mx.arange(W).astype(mx.float32) - 5,
    )
    y = np.array(apply_axial_rope(x, t_pos, h_pos, w_pos, split, inv))
    xn = np.array(x)
    dt, dh, dw = split
    # T block: rotate along axis T
    ref_t = _rot_ref(np.moveaxis(xn[..., :dt], 1, -2), np.array(t_pos), np.array(inv[0]))
    assert np.allclose(y[..., :dt], np.moveaxis(ref_t, -2, 1), atol=1e-5)
    ref_h = _rot_ref(np.moveaxis(xn[..., dt : dt + dh], 2, -2), np.array(h_pos), np.array(inv[1]))
    assert np.allclose(y[..., dt : dt + dh], np.moveaxis(ref_h, -2, 2), atol=1e-5)
    ref_w = _rot_ref(np.moveaxis(xn[..., dt + dh :], 3, -2), np.array(w_pos), np.array(inv[2]))
    assert np.allclose(y[..., dt + dh :], np.moveaxis(ref_w, -2, 3), atol=1e-5)


def test_axial_rope_keeps_bf16_dtype():
    x = mx.random.normal((1, 2, 2, 2, 1, 8)).astype(mx.bfloat16)
    split = rope_dim_split(8)
    y = apply_axial_rope(
        x,
        mx.arange(2).astype(mx.float32),
        mx.arange(2).astype(mx.float32),
        mx.arange(2).astype(mx.float32),
        split,
        tuple(inv_freqs(d) for d in split),
    )
    assert y.dtype == mx.bfloat16
