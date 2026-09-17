"""Decoder selection, size guard and CLI plumbing for the diffusion video decoder (no weights)."""

from __future__ import annotations

import mlx.core as mx
import pytest

from ltx_pipelines_mlx._base import BasePipeline
from ltx_pipelines_mlx.utils import blocks as B  # noqa: N812


def test_video_decoder_block_defaults_to_conv_and_validates_choice(tmp_path):
    assert B.VideoDecoder(tmp_path).video_decoder == "conv"
    with pytest.raises(ValueError, match="video_decoder"):
        B.VideoDecoder(tmp_path, video_decoder="magic")


def test_diffusion_choice_requires_the_av_weights(tmp_path):
    vd = B.VideoDecoder(tmp_path, video_decoder="diffusion")
    with pytest.raises(FileNotFoundError, match=r"vae_decoder_av\.safetensors"):
        vd.load()


def test_stage5_token_estimate_and_guard(monkeypatch):
    # 512x768x49 -> latent (7, 16, 24) -> stage-5 grid (8*7-7=49) x (16*8=128) x (24*8=192) = 1_204_224
    assert B._DiffusionVideoDecoder.estimate_stage5_tokens((1, 128, 7, 16, 24)) == 49 * 128 * 192
    monkeypatch.setenv(B.DIFFVAE_MAX_TOKENS_ENV, "1000")
    with pytest.raises(ValueError, match="LTX2_DIFFVAE_MAX_TOKENS"):
        B._DiffusionVideoDecoder.check_size((1, 128, 7, 16, 24))
    monkeypatch.delenv(B.DIFFVAE_MAX_TOKENS_ENV)
    B._DiffusionVideoDecoder.check_size((1, 128, 7, 16, 24))  # default 2.5 M passes


def test_diffusion_decode_and_stream_uses_shared_ffmpeg_plumbing(monkeypatch, tmp_path):
    seen = {}

    class _Dec:
        def tiled_decode(self, latent, tiling=None, *, seed=0):
            yield mx.zeros((1, 3, 9, 64, 96))

    def _fake_ffmpeg_sink(cmd):
        seen["cmd"] = cmd
        return _FakeSink()

    monkeypatch.setattr(B, "_ffmpeg_sink", _fake_ffmpeg_sink)
    monkeypatch.setattr(
        B, "stream_chunks_to_ffmpeg", lambda chunks, proc: seen.setdefault("frames", sum(c.shape[2] for c in chunks))
    )
    wrapper = B._DiffusionVideoDecoder(_Dec())
    wrapper.decode_and_stream(mx.zeros((1, 128, 2, 2, 3)), str(tmp_path / "o.mp4"), frame_rate=24.0, audio_path=None)
    assert (seen["frames"] == 9 and "64x96" in " ".join(seen["cmd"])) or "96x64" in " ".join(seen["cmd"])


class _FakeSink:
    def __enter__(self):
        class _P:
            stdin = None

        return _P()

    def __exit__(self, *a):
        return False


def test_base_pipeline_forwards_video_decoder_to_the_block(monkeypatch):
    p = BasePipeline.__new__(BasePipeline)
    assert p.video_decoder == "conv"
    p.verbose = False
    p.generate_audio = False
    p.video_decoder = "diffusion"
    calls = {}

    class _Blk:
        video_decoder = "conv"

        def load(self):
            calls["loaded_with"] = self.video_decoder

    p.video_decoder_block = _Blk()
    p.audio_decoder_block = None
    p._load_decoders()
    assert calls["loaded_with"] == "diffusion"


def _parse(*extra):
    from ltx_pipelines_mlx.cli import _build_parser

    return _build_parser().parse_args(
        ["generate", "-p", "x", "-o", "o.mp4", "--frame-rate", "24", "-f", "9", "--distilled", *extra]
    )


def test_cli_flag_defaults_and_parses():
    assert _parse().video_decoder == "conv"
    assert _parse("--video-decoder", "diffusion").video_decoder == "diffusion"


@pytest.mark.parametrize("mode", ["--distilled", "--one-stage", "--two-stage", "--two-stages-hq"])
def test_cli_flag_reaches_every_generate_mode(monkeypatch, mode):
    seen = {}

    class _FakePipe:
        def __init__(self, *a, **k):
            pass

        def generate_and_save(self, **kwargs):
            seen["video_decoder"] = getattr(self, "video_decoder", "UNSET")

    import ltx_pipelines_mlx.distilled as d
    import ltx_pipelines_mlx.ti2vid_one_stage as o
    import ltx_pipelines_mlx.ti2vid_two_stages as t
    import ltx_pipelines_mlx.ti2vid_two_stages_hq as hq

    monkeypatch.setattr(d, "DistilledPipeline", _FakePipe)
    monkeypatch.setattr(o, "TI2VidOneStagePipeline", _FakePipe)
    monkeypatch.setattr(t, "TI2VidTwoStagesPipeline", _FakePipe)
    monkeypatch.setattr(hq, "TI2VidTwoStagesHQPipeline", _FakePipe)
    from ltx_pipelines_mlx.cli import _build_parser, _cmd_generate

    args = _build_parser().parse_args(
        [
            "generate",
            "-p",
            "x",
            "-o",
            "o.mp4",
            "--frame-rate",
            "24",
            "-f",
            "9",
            mode,
            "--video-decoder",
            "diffusion",
            "--quiet",
        ]
    )
    _cmd_generate(args)
    assert seen["video_decoder"] == "diffusion"
