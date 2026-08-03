"""ChimeraForge CLI：五阶段流水线命令 + demo 端到端冒烟测试。

用法示例：
  python -m chimeraforge genkey
  python -m chimeraforge compile --content secret.txt --out artifacts/model.pt
  python -m chimeraforge compress --model artifacts/model.pt --out artifacts/payload.bin
  python -m chimeraforge embed --payload artifacts/payload.bin --cover cover.mp4 --out artifacts/stego.cfb
  python -m chimeraforge extract --stego artifacts/stego.cfb --out artifacts/payload2.bin
  python -m chimeraforge reconstruct --payload artifacts/payload2.bin --out artifacts/secret.out
  python -m chimeraforge demo --content secret.txt --outdir artifacts
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

from omegaconf import OmegaConf

from chimeraforge import keys
from chimeraforge.stages import compile_ as stage_compile
from chimeraforge.stages import compress as stage_compress
from chimeraforge.stages import embed as stage_embed
from chimeraforge.stages import extract as stage_extract
from chimeraforge.stages import reconstruct as stage_reconstruct

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def load_config(path: str | None) -> OmegaConf:
    cfg = OmegaConf.load(str(DEFAULT_CONFIG))
    if path:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    return cfg


def resolve_master_key(cfg, key_arg: str | None) -> bytes:
    if key_arg:
        return keys.normalize_master_key(key_arg)
    env = os.environ.get("CHIMERAFORGE_MASTER_KEY")
    if env:
        return keys.normalize_master_key(env)
    return keys.normalize_master_key(cfg.master_key)


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="额外 YAML 配置（合并覆盖 default.yaml）")
    p.add_argument("--key", help="主密钥（优先于环境变量与配置）")


def cmd_compile(args) -> int:
    cfg = load_config(args.config)
    info = stage_compile.run(args.content, args.out, cfg, resolve_master_key(cfg, args.key), progress=args.verbose)
    print(f"[compile] 模型已写入 {args.out}（原始 {info['orig_size']} B）")
    print(f"          epochs={info['stats']['epochs']} loss={info['stats']['final_loss']:.3e} "
          f"字节准确率={info['stats']['byte_accuracy']:.4f} 耗时={info['stats']['seconds']:.1f}s")
    return 0


def cmd_compress(args) -> int:
    cfg = load_config(args.config)
    info = stage_compress.run(args.model, args.out, cfg, resolve_master_key(cfg, args.key))
    print(f"[compress] 载荷已写入 {args.out}（{info['size']} B, ECC={info['ecc']}）")
    return 0


def cmd_embed(args) -> int:
    cfg = load_config(args.config)
    info = stage_embed.run(args.payload, args.cover, args.out, cfg, resolve_master_key(cfg, args.key))
    print(f"[embed] 后端={info['backend']} 隐写文件已写入 {args.out}（{info['size']} B）")
    return 0


def cmd_extract(args) -> int:
    cfg = load_config(args.config)
    info = stage_extract.run(args.stego, args.out, cfg, resolve_master_key(cfg, args.key))
    print(f"[extract] 后端={info['backend']} 载荷已提取到 {args.out}（{info['size']} B）")
    return 0


def cmd_reconstruct(args) -> int:
    cfg = load_config(args.config)
    info = stage_reconstruct.run(args.payload, args.out, cfg, resolve_master_key(cfg, args.key))
    print(f"[reconstruct] 内容已重建到 {args.out}（{info['got_size']} B / 期望 {info['orig_size']} B）")
    return 0


def cmd_demo(args) -> int:
    cfg = load_config(args.config)
    master_key = resolve_master_key(cfg, args.key)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, payload, extracted = outdir / "model.pt", outdir / "payload.bin", outdir / "payload.extracted.bin"
    stego = outdir / "stego.cfb"
    content_out = outdir / "content.out.bin"

    step("compile", stage_compile.run(args.content, str(model), cfg, master_key, progress=args.verbose))
    step("compress", stage_compress.run(str(model), str(payload), cfg, master_key))
    step("embed", stage_embed.run(str(payload), args.cover, str(stego), cfg, master_key))
    step("extract", stage_extract.run(str(stego), str(extracted), cfg, master_key))
    step("reconstruct", stage_reconstruct.run(str(extracted), str(content_out), cfg, master_key))

    original = Path(args.content).read_bytes()
    rebuilt = content_out.read_bytes()
    ok = original == rebuilt
    print("-" * 52)
    print(f"  SHA-256(原始)   : {hashlib.sha256(original).hexdigest()[:32]}...")
    print(f"  SHA-256(重建)   : {hashlib.sha256(rebuilt).hexdigest()[:32]}...")
    print(f"  模型大小        : {model.stat().st_size / 1024:.1f} KB (fp32 原始)")
    print(f"  载荷大小        : {payload.stat().st_size} B (int{cfg.compress.bits} 量化 + zlib + 头)")
    print(f"  隐写文件大小    : {stego.stat().st_size} B")
    print(f"  roundtrip       : {'PASS' if ok else 'FAIL'}")
    print("-" * 52)
    return 0 if ok else 1


def step(name: str, info: dict) -> None:
    size = info.get("size")
    tail = f" ({size} B)" if size else ""
    print(f"  [{name:10s}] {info.get('stage', name)}{tail}")


def cmd_genkey(args) -> int:
    print(keys.gen_master_key())
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chimeraforge", description="模型即信息：编译→压缩→隐写→提取→重建")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("compile", help="内容字节 → INR 权重")
    c.add_argument("--content", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--verbose", action="store_true")
    _add_common(c)
    c.set_defaults(func=cmd_compile)

    c = sub.add_parser("compress", help="INR 权重 → 认证载荷")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    _add_common(c)
    c.set_defaults(func=cmd_compress)

    c = sub.add_parser("embed", help="载荷 → 载体视频")
    c.add_argument("--payload", required=True)
    c.add_argument("--cover", help="载体文件（stub 后端可省略）")
    c.add_argument("--out", required=True)
    _add_common(c)
    c.set_defaults(func=cmd_embed)

    c = sub.add_parser("extract", help="隐写视频 → 载荷")
    c.add_argument("--stego", required=True)
    c.add_argument("--out", required=True)
    _add_common(c)
    c.set_defaults(func=cmd_extract)

    c = sub.add_parser("reconstruct", help="载荷 → 原始内容")
    c.add_argument("--payload", required=True)
    c.add_argument("--out", required=True)
    _add_common(c)
    c.set_defaults(func=cmd_reconstruct)

    c = sub.add_parser("demo", help="端到端冒烟：compile→compress→embed→extract→reconstruct")
    c.add_argument("--content", required=True)
    c.add_argument("--cover", help="载体文件（可选）")
    c.add_argument("--outdir", default="artifacts")
    c.add_argument("--verbose", action="store_true")
    _add_common(c)
    c.set_defaults(func=cmd_demo)

    c = sub.add_parser("genkey", help="生成随机主密钥（hex）")
    c.set_defaults(func=cmd_genkey)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as e:  # noqa: BLE001 - CLI 顶层统一错误输出
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
