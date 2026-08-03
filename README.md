# ChimeraForge 🔥 模型即信息

> 不藏数据，藏"种子"——用一个小型神经网络的权重替代秘密数据本体，
> 再将其隐写进普通视频。持有主密钥者，可从视频中重建出原始内容。

ChimeraForge 是一个 **模型即信息（Model-as-information）** 隐写框架：
秘密内容被"编译"成一个 1-D 隐式神经表示（INR）的权重（即"种子模型"），
经 int8 量化压缩后，借助 LF-VSN（CVPR 2023 可逆神经网络视频隐写）嵌入载体视频。
提取端反向运行同一套预训练权重，重建出种子模型，再前向推理恢复原始内容。

## 五阶段流水线

```
秘密内容 ─compile→ INR权重(种子) ─compress→ 认证载荷 ─embed→ 载体视频
秘密内容 ←reconstruct← INR权重     ←decompress← 认证载荷 ←extract← 载体视频
```

| 阶段 | 作用 | 技术 |
|---|---|---|
| `compile` | 内容字节 → 1-D INR 权重（种子模型） | NeRV 风格 MLP：位置编码 + GELU + FiLM 逐层调制（主密钥派生） |
| `compress` | 权重 → 认证载荷 | int8 量化感知训练（QAT）+ 熵编码（zlib）+ MAC/HMAC + Reed-Solomon ECC（可选） |
| `embed` | 载荷 → 载体视频 | LF-VSN（可逆神经网络）隐藏秘密帧；比特平面冗余编码抗有损重建 |
| `extract` | 载体视频 → 载荷 | LF-VSN 逆向数据流 + 冗余多数投票阈值解码 |
| `reconstruct` | 载荷 → 原始内容 | MAC 校验 → 权重反量化 → INR 按索引前向 → SHA-256 校验 |

安全性来自两级密钥：主密钥派生 FiLM 调制参数注入 INR（无密钥无法重建内容），
载荷层另有 HMAC 完整性校验与（可选）Reed-Solomon 纠错以对抗视频平台二次转码。

## 技术栈

- Python 3.10+ · PyTorch 2.x · OmegaConf · NumPy
- LF-VSN（`MC-E/LF-VSN`，CVPR 2023，git submodule + Google Drive 预训练权重）
- 可选：`gdown`（权重下载）、`opencv-python`/`scikit-image`（视频帧 I/O）、`reedsolo`（ECC）

## 安装

```bash
# 1. 克隆并初始化 LF-VSN submodule
git clone --recurse-submodules https://github.com/your/chimeraforge
cd chimeraforge
git submodule update --init third_party/LF-VSN

# 2. 安装依赖
pip install -e ".[dev]"
# 需要视频后端/ECC 时：
pip install -e ".[lfvsn,ecc]"

# 3. 下载 LF-VSN 预训练权重（单视频隐藏模式）
python scripts/download_lfvsn_weights.py --mode 1 --out third_party/LF-VSN/pretrained
```

## 快速开始

```bash
# 端到端冒烟（默认 stub 后端，无需 GPU）
python -m chimeraforge demo --content secret.txt --outdir artifacts

# 五阶段手动执行
python -m chimeraforge genkey
python -m chimeraforge compile   --content secret.txt --out artifacts/model.pt
python -m chimeraforge compress  --model artifacts/model.pt --out artifacts/payload.bin
python -m chimeraforge embed     --payload artifacts/payload.bin --cover cover.mp4 --out artifacts/stego.cfb
python -m chimeraforge extract   --stego artifacts/stego.cfb --out artifacts/payload2.bin
python -m chimeraforge reconstruct --payload artifacts/payload2.bin --out artifacts/secret.out
```

主密钥优先级：`--key` > 环境变量 `CHIMERAFORGE_MASTER_KEY` > `configs/default.yaml`。

切换到 LF-VSN 真实隐写后端：编辑 `configs/default.yaml`，将 `stego.backend` 改为 `lfvsn`
（需预训练权重就位，参见 `stego.lfvsn` 配置段）。

## 配置

`configs/default.yaml`（OmegaConf，可用 `--config xxx.yaml` 合并覆盖）：

```yaml
inr:      # 种子模型结构 / 训练轮数 / int8 QAT 微调轮数
compress: # 量化位宽（8=int8 QAT，16=int16 PTQ）/ 熵编码器
stego:    # 后端（stub|lfvsn）、LF-VSN 权重路径、冗余度、合成封面分辨率
ecc:      # Reed-Solomon 纠错开关与符号数
```

## 测试

```bash
python -m pytest -q        # 73 项：单元 + 五阶段 roundtrip + QAT 无损 + LF-VSN 错误路径
```

QAT 链路保证字节级无损：余弦退火微调 + 训练结束将权重显式拉回 int8 格点，
`pack_model`/`unpack_model` 与训练前向位级一致。

## 项目结构

```
chimeraforge/
  inr/            # 1-D INR：模型、训练、QAT、量化打包
  stego/          # 隐写后端：stub 占位、LF-VSN 封装、载荷编解码
  stages/         # compile / compress / embed / extract / reconstruct
  keys.py         # 主密钥 KDF（FiLM 种子 / 隐写密钥 / MAC 密钥）
  payload.py      # 载荷格式（CFP1 头 + MAC + INR 结构哈希）
  codecs.py       # Reed-Solomon 纠错
  cli.py          # 五命令 + demo + genkey
configs/default.yaml
scripts/download_lfvsn_weights.py
third_party/LF-VSN   # git submodule（CVPR 2023 官方实现）
tests/
```

## 许可证

本项目 `chimeraforge` 包本体以 **MIT License** 发布（见 [LICENSE](LICENSE)）。
第三方组件（LF-VSN 等）的许可证状态与使用边界见 [NOTICE.md](NOTICE.md)：

- **LF-VSN 及其预训练权重无开源许可证（保留所有权利）**，以 git submodule 引用、
  仅限研究用途；公开分发或商用前请联系作者授权；
- `dwt`/`iwt` 为标准 Haar 小波独立实现，已验证与官方推理逐位一致；
- 其余 pip 依赖均为宽松许可证（BSD/MIT/Apache-2.0）。

## 里程碑状态

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 工程骨架 | repo 结构、LF-VSN submodule、CLI、roundtrip 冒烟 | ✅ 完成 |
| M1 MVP 全流程 | 文本/小图 → INR → int8 QAT → LF-VSN 嵌入 → 提取 → 重建 | 🚧 代码完成（权重后补，待 720p 端到端验收） |
| M2 内容升级 | 短视频作为秘密；载体 720p | 待办 |
| M3 研究课题 | 高熵数据表示、抗压缩鲁棒性、多秘密多接收者 | 待办 |

验收标准：种子模型 < 2MB、重建 SHA-256 与原文一致、720p 载体端到端可跑。

## 致谢

- LF-VSN: *Large-Capacity and Flexible Video Steganography via Invertible Neural Network*（CVPR 2023），[MC-E/LF-VSN](https://github.com/MC-E/LF-VSN)
- 技术路线参考：NeRV / E-NeRV / TinyNeRV（INR 与模型压缩）、NeR-VCP（模型即密文）
