# 第三方组件与许可证声明

本项目（`chimeraforge` 包本体）以 **MIT License** 发布（见 [LICENSE](LICENSE)）。
以下为本项目引入或参考的第三方组件及其许可证状态。**本声明不授予任何第三方
组件超出其自身许可范围的额外权利。**

## 直接引入的组件（运行时依赖）

| 组件 | 来源 | 许可证 | 使用方式与说明 |
|---|---|---|---|
| LF-VSN | [MC-E/LF-VSN](https://github.com/MC-E/LF-VSN)（CVPR 2023） | **无开源许可证（保留所有权利）** | 以 git submodule 引用（`third_party/LF-VSN`），仅运行时加载其网络模块与预训练权重。官方仓库与权重均未附带 LICENSE。**研究用途默认许可；公开分发或商业使用前请联系作者取得授权**（作者邮箱见官方 README：eechongm@gmail.com）。 |
| MIMO-VRN | [ding3820/MIMO-VRN](https://github.com/ding3820/MIMO-VRN)（CVPR 2021） | **无开源许可证（保留所有权利）** | LF-VSN 的代码基础（其 Acknowledgement 声明），未直接引入本项目仓库。 |
| TinyNeRV-Implementation | [HannanAkhtar/TinyNeRV-Implementation](https://github.com/HannanAkhtar/TinyNeRV-Implementation) | MIT | 仅参考其 QAT/蒸馏方案思想，无代码复制。 |

## 思想/风格参考（无代码复制）

| 组件 | 许可证 | 说明 |
|---|---|---|
| NeRV（haochen-rye/NeRV） | 无开源许可证 | 参考其 MLP 风格 INR 结构设计，按本项目 1-D 文件信号重新实现 |
| E-NeRV（kyleleey/E-NeRV） | 无开源许可证 | 同上 |
| NeR-VCP | 无公开代码 | 理论参考（模型即密文） |

## Python 运行时依赖（pip 安装，不并入本仓库分发物）

| 依赖 | 许可证 |
|---|---|
| torch / numpy / omegaconf | BSD-3-Clause |
| gdown / reedsolo / pytest | MIT |
| opencv-python | Apache-2.0 |
| scikit-image | BSD-3-Clause |

## 预训练权重

LF-VSN 预训练权重托管于 Google Drive，**未附带明确许可证**，请仅用于研究目的。

## 重写声明

本项目 `chimeraforge/stego/lfvsn_model.py` 中的 `dwt`/`iwt` 为标准 Haar
小波算法的独立实现（已通过数值验证与 LF-VSN 官方推理逐位一致），未复制
官方 `common.py` 代码；其余对 LF-VSN 的调用以 submodule 方式完成。
