"""LF-VSN 后端（M1 接入，INN 视频隐写）。

依赖：`third_party/LF-VSN`（git submodule）+ 预训练权重（scripts/download_lfvsn_weights.py）。

媒体层流程（每个 GOP=3 帧转移 1 个中心秘密帧，官方训练即按中心帧监督）：
  embed:   载荷 →(媒体层 ECC→密钥流 XOR→bit 编码)→ 秘密帧 → 逐帧 hide_gop → 隐写帧 → 写盘
  extract: 隐写帧 → 逐帧 reveal_gop → 秘密帧 →(bit 解码→XOR→ECC)→ 载荷

载体/隐写格式：
  - 载体：PNG 帧目录 或 视频文件（cv2 读取）
  - 隐写：默认 PNG 帧目录（无损，保证提取可靠）；扩展名为视频时写 MP4（有损，实验性）
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from chimeraforge import codecs
from . import payload_codec
from .base import StegoBackend

# 文件名遵循 scripts/download_lfvsn_weights.py 的统一命名约定
_WEIGHT_FILES = {n: f"lfvsn_mode{n}.pth" for n in range(1, 8)}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_PNG_EXTS = {".png"}


def _synthetic_cover(n_frames: int, height: int, width: int, seed: int = 7) -> list[np.ndarray]:
    """确定性合成封面：移动渐变 + 轻微噪声（无载体时的兜底，测试可复现）。"""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    frames = []
    for k in range(n_frames):
        t = k / max(n_frames, 1)
        r = 128.0 + 100.0 * np.sin(2 * np.pi * (xx / width * 0.5 + t))
        g = 128.0 + 100.0 * np.sin(2 * np.pi * (yy / height * 0.4 - t))
        b = 128.0 + 100.0 * np.cos(2 * np.pi * ((xx + yy) / (width + height) + t))
        frame = np.stack([b, g, r], axis=-1) + rng.normal(0.0, 3.0, (height, width, 3))
        frames.append(np.clip(frame, 0, 255).astype(np.uint8))
    return frames


def _read_frames(source: str | Path) -> list[np.ndarray]:
    """读取帧序列（BGR uint8）。支持 PNG 目录或视频文件。"""
    src = Path(source)
    if src.is_dir():
        import cv2

        paths = sorted(p for p in src.iterdir() if p.suffix.lower() in _PNG_EXTS)
        frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
        return [f for f in frames if f is not None]
    if src.suffix.lower() in _VIDEO_EXTS:
        import cv2

        cap = cv2.VideoCapture(str(src))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames
    raise ValueError(f"不支持的载体格式: {src}（支持 PNG 目录或视频文件）")


def _write_frames(frames: list[np.ndarray], dest: str | Path) -> tuple[str, int]:
    """写帧序列。目录 → PNG（无损）；视频扩展名 → MP4（有损）。返回 (格式, 字节数)。"""
    dst = Path(dest)
    if dst.suffix.lower() in _VIDEO_EXTS:
        import cv2

        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()
        return "mp4", dst.stat().st_size
    dst.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        import cv2

        cv2.imwrite(str(dst / f"frame_{i:05d}.png"), f)
    total = sum(p.stat().st_size for p in dst.iterdir())
    return "png", total


def _to_rgb_float(bgr_frames: list[np.ndarray], pad_h: int, pad_w: int) -> torch.Tensor:
    """BGR uint8 帧 → (T, 3, H, W) RGB float [0,1]，边界复制补到偶数边。"""
    rgb = np.stack([f[:, :, ::-1] for f in bgr_frames])  # (T,H,W,3) RGB
    t = torch.from_numpy(rgb.astype(np.float32) / 255.0)
    t = t.permute(0, 3, 1, 2).contiguous()
    if t.shape[2] != pad_h or t.shape[3] != pad_w:
        t = torch.nn.functional.pad(t, (0, pad_w - t.shape[3], 0, pad_h - t.shape[2]), mode="replicate")
    return t


def _to_bgr_uint8(rgb_float: torch.Tensor) -> np.ndarray:
    """(3, H, W) RGB float [0,1] → (H, W, 3) BGR uint8。"""
    arr = (rgb_float.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)
    return arr.transpose(1, 2, 0)[:, :, ::-1].copy()


def _gop3(t: torch.Tensor, k: int) -> torch.Tensor:
    """取以 k 为中心的长度 3 序列（边界复制）。t: (T, 3, H, W)。"""
    if k == 0:
        return torch.cat([t[:1].repeat(2, 1, 1, 1), t[:2]], 0)
    if k == t.shape[0] - 1:
        return torch.cat([t[-2:], t[-1:].repeat(2, 1, 1, 1)], 0)
    return t[k - 1 : k + 2]


class LFVSNBackend(StegoBackend):
    name = "lfvsn"

    def __init__(
        self,
        weight_dir: str | Path,
        mode: int = 1,
        frame_size: int = 256,
        redundancy: int = 4,
        device: str = "auto",
        ecc_nsym: int = 0,
    ):
        self.weight_dir = Path(weight_dir)
        self.mode = mode
        self.frame_size = frame_size
        self.redundancy = max(1, int(redundancy))
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.ecc_nsym = int(ecc_nsym)
        self._net = None
        self._check()

    def _check(self) -> None:
        weight = self.weight_dir / _WEIGHT_FILES[self.mode]
        repo = Path(__file__).resolve().parents[2] / "third_party" / "LF-VSN"
        if not repo.exists():
            raise RuntimeError(
                "未找到 LF-VSN 源码：请执行 `git submodule update --init third_party/LF-VSN`"
            )
        if not weight.exists():
            raise RuntimeError(
                f"未找到 LF-VSN 预训练权重 {weight.name}：\n"
                f"  运行 `python scripts/download_lfvsn_weights.py --mode {self.mode} "
                f"--out {self.weight_dir}`"
            )

    def _load(self) -> torch.nn.Module:
        if self._net is None:
            self._check()
            from .lfvsn_model import load_lfvsn

            try:
                self._net = load_lfvsn(self.weight_dir / _WEIGHT_FILES[self.mode], self.device)
            except Exception as e:
                raise RuntimeError(
                    f"LF-VSN 权重加载失败（{self.weight_dir / _WEIGHT_FILES[self.mode]}）：\n{e}"
                ) from e
        return self._net

    def embed(self, data: bytes, cover_path: Path | None, out_path: Path, key: bytes) -> dict:
        net = self._load()
        if self.ecc_nsym:
            data = codecs.encode(data, nsym=self.ecc_nsym)

        if cover_path is None:
            cover = None
            height = width = self.frame_size
        else:
            cover = _read_frames(cover_path)
            height, width = cover[0].shape[:2]
        pad_h, pad_w = height + height % 2, width + width % 2

        secret = payload_codec.encode(data, key, pad_h, pad_w, self.redundancy)  # (N,3,H,W)
        n = secret.shape[0]
        if cover is None:
            cover = _synthetic_cover(n, pad_h, pad_w)
        cover_t = _to_rgb_float(cover, pad_h, pad_w)
        if cover_t.shape[0] < n:
            raise ValueError(f"载体帧数不足：需 {n} 帧，实际仅 {cover_t.shape[0]} 帧")

        from .lfvsn_model import hide_gop

        stego = []
        for k in range(n):
            host_g = _gop3(cover_t, k).to(self.device)
            secret_g = _gop3(secret, k).to(self.device)
            out = hide_gop(net, host_g, secret_g)  # (3, pad_h, pad_w) [0,1]
            stego.append(_to_bgr_uint8(out))
        fmt, size = _write_frames(stego, out_path)

        return {
            "backend": self.name,
            "size": size,
            "frames": n,
            "resolution": f"{pad_w}x{pad_h}",
            "redundancy": self.redundancy,
            "stego_fmt": fmt,
        }

    def extract(self, stego_path: Path, out_path: Path, key: bytes) -> int:
        net = self._load()
        frames = _read_frames(stego_path)
        if not frames:
            raise ValueError("隐写载体为空或无法读取")
        h, w = frames[0].shape[:2]
        stego_t = _to_rgb_float(frames, h + h % 2, w + w % 2)

        from .lfvsn_model import reveal_gop

        recovered = []
        for k in range(stego_t.shape[0]):
            rec = reveal_gop(net, stego_t[k].to(self.device))  # (3, H, W) [0,1]
            recovered.append(rec.cpu())
        secret_t = torch.stack(recovered)

        payload = payload_codec.decode(secret_t, key, self.redundancy)
        if self.ecc_nsym:
            payload = codecs.decode(payload, nsym=self.ecc_nsym)
        Path(out_path).write_bytes(payload)
        return len(payload)
