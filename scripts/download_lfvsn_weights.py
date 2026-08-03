"""下载 LF-VSN 官方预训练权重（Google Drive），统一命名为 lfvsn_mode{n}.pth。

用法：
    python scripts/download_lfvsn_weights.py --mode 1 --out third_party/LF-VSN/pretrained

依赖：优先使用 gdown（pip install gdown）；不可用时打印手动下载指引。
"""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

# 官方 README 中的 Google Drive 文件 ID（1~7 个秘密视频隐藏模式）
MODE_IDS = {
    1: "1aEMZaigkMd2NUNXnOu2r0oa5IuLPCtTh",
    2: "1Yd7tK9Y-J4fkXoL-5u8VifEVsW7OmZN0",
    3: "1oeDDzkYMZ6tKpPnIUwSI2v_Rbn7vLQJo",
    4: "1kyMKdfAG_gq6ArWChv6ZMLBsqT-QpS9j",
    5: "1OlTL6_ZgsThPeYfxbpGrGvNoaisqThq2",
    6: "1dr-ZIL-VP0ol4fRO7bGZYQoxRetA-GXW",
    7: "178cqpz_vS-mPlYwLuZP2qFc7pV7vrXrr",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="下载 LF-VSN 预训练权重")
    ap.add_argument("--mode", type=int, choices=sorted(MODE_IDS), required=True, help="隐藏模式（秘密视频数量 1~7）")
    ap.add_argument("--out", default="third_party/LF-VSN/pretrained")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"lfvsn_mode{args.mode}.pth"
    if target.exists():
        print(f"已存在：{target}，跳过。")
        return 0

    drive_url = f"https://drive.google.com/file/d/{MODE_IDS[args.mode]}/view"
    try:
        import gdown  # type: ignore
    except ImportError:
        print("未安装 gdown，请手动下载：")
        print(f"  打开 {drive_url} 下载后，解压并把 .pth 文件重命名为 {target.name} 放入 {out_dir}")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp_zip = Path(td) / "lfvsn.zip"
        print(f"下载中（mode={args.mode}）...")
        gdown.download(f"https://drive.google.com/uc?id={MODE_IDS[args.mode]}", str(tmp_zip), quiet=False)
        with zipfile.ZipFile(tmp_zip) as z:
            pth_names = [n for n in z.namelist() if n.endswith(".pth")]
            if not pth_names:
                raise RuntimeError("压缩包中未找到 .pth 权重文件")
            with z.open(pth_names[0]) as src, open(target, "wb") as dst:
                dst.write(src.read())
    print(f"完成：{target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
