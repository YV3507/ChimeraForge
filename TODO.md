# ChimeraForge TODO

按优先级排序；`[ ]` = 待办，`[x]` = 已完成。

## M1 MVP（当前里程碑）

- [ ] 下载 LF-VSN 预训练权重（`third_party/LF-VSN/pretrained/lfvsn_mode1.pth`）——**暂缓**：网络连接不佳；脚本已支持（`scripts/download_lfvsn_weights.py`）
- [ ] M1 端到端验收（依赖权重就绪后）：真实 LF-VSN 后端下 720p 载体跑通五阶段
  - [ ] 种子模型 < 2MB（int8 已实测 294KB，需在端到端链路复核）
  - [ ] 重建 SHA-256 与原文一致
  - [ ] 媒体层冗余度 / ECC 参数在真实有损重建下的错误率实测（`disagreement_rate` 诊断）
- [ ] MP4 载体端到端路径（媒体层 I/O 已验证；真实后端验证随权重就绪后完成）
- [x] 五阶段流水线代码（compile → compress → embed → extract → reconstruct）
- [x] int8 QAT 无损链路（余弦退火 + 格点 snap）
- [x] LF-VSN 封装（basicsr shim、DWT/IWT 独立实现、GOP 处理）
- [x] 载荷↔秘密视频张量编解码（比特平面 + 冗余 + 阈值投票）
- [x] 媒体层 I/O：PNG 目录（无损）/ MP4（有损）roundtrip 测试
- [x] 83 项测试全绿

## M2 内容升级

- [ ] 短视频作为秘密内容（INR 需支持更大输入规模）
- [ ] 载体 720p / 更长片段；PSNR 达标评估
- [ ] 视频平台二次转码（H.264 重压缩）鲁棒性实测

## M3 研究课题

- [ ] 高熵数据表示（哈希网格 / 可逆容器），评估模型大小-熵的权衡
- [ ] 抗压缩鲁棒性训练（模拟压缩环节加入隐写训练）
- [ ] 多秘密 / 多接收者（LF-VSN 密钥控制的多视频模式 2~7）

## 工程杂项

- [ ] `scripts/download_lfvsn_weights.py` 支持 HTTP 代理 / 镜像源
- [ ] LF-VSN 权重缺失时给出更细的恢复指引（当前为清晰错误路径）
- [x] CI 配置（GitHub Actions：submodule 初始化 + pytest，Python 3.10/3.12 矩阵）
- [x] GOP 边界帧（首/尾复制补齐）、合成封面确定性测试
- [x] CLI 各子命令端到端覆盖（11 项，含错误路径）
