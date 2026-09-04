# Matching Clothes｜服装找款与配货标注工具

[简体中文](README.md) | [English](README.en.md) | [한국어](README.ko.md)

一个面向 Windows 的本地服装找款工具：从可变款式图库中为门店实拍图搜索同款，读取红色尺码/颜色标注，经人工确认后生成标注结果、任务明细和补货汇总。

项目默认离线运行，图库、实拍图和结果不会上传。AI 负责给候选排序，最终款式由人确认，适合服装找款、补货和库存整理。

## 主要功能

- 四视角视觉匹配：整件、中央图案、上半身和小胸标。
- OCR 文字与轻量纹理融合，颜色只作为弱证据，支持同款不同色。
- 优先读取实拍图中的红色颜色、尺码和数量标注。
- 每张图默认显示 20 个候选，可继续展开或手动选择。
- 两个预览区支持滚轮缩放、左键拖动和双击复位。
- 人工确认结果可作为当前图库的本地反馈样本，不跨图库串用。
- 输出任务总览、逐图结果、`任务明细.csv` 和 `补货汇总.csv`。
- 原图库和实拍图只读，所有红字均写入结果副本。

## 快速开始

### 1. 环境

- Windows 10/11
- Python 3.10 或更高版本
- 推荐 8 GB 以上内存

```powershell
git clone https://github.com/wei121212121/Matching_Clothes.git
cd Matching_Clothes\clothing_matcher_v8_ui_alt
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

没有 ONNX 模型时程序仍可使用内置轻量视觉特征启动，但候选精度会明显下降。项目把约 94 MB 的兼容模型作为独立的 [GitHub Release 资源](https://github.com/wei121212121/Matching_Clothes/releases/tag/model-v1) 提供，避免把大文件写入 Git 历史。可在仓库根目录运行以下命令自动下载、校验 SHA-256，并放到正确位置：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\download_model.ps1
```

模型来源、许可证、文件校验值和手动安装方法见 [模型说明](docs/模型说明.md) 与 [第三方声明](THIRD_PARTY_NOTICES.md)。

### 2. 用公开示例试跑

仓库包含一套完全合成的 [演示数据](examples/README.md)，不含真实业务图片或私人信息：

1. 把 `examples/style_library` 选为“款式图库”。
2. 另建一个空文件夹作为“结果目录”，点击“建立/更新索引”。
3. 导入 `examples/store_photos`，点击“分析新照片（F5）”。
4. 将候选结果与 `examples/expected_matches.csv` 对照。

### 3. 第一次使用

1. 选择“款式图库”和“结果目录”。
2. 点击“建立/更新索引”。
3. 导入实拍图，或从资源管理器复制后在左侧按 `Ctrl+V`。
4. 点击“分析新照片（F5）”。
5. 检查实拍图、候选图、颜色和尺码，确认正确候选。
6. 全部确认后点击“导出本批全部结果（Ctrl+E）”。

完整步骤、界面操作和故障排查见 [使用手册](docs/使用手册.md)。

## 常用快捷键

| 快捷键 | 操作 |
| --- | --- |
| `F5` | 分析新照片 |
| `Ctrl+Enter` | 确认当前候选 |
| `Ctrl+→` | 下一张待确认 |
| `Ctrl+←` | 上一张 |
| `Ctrl+E` | 导出本批结果 |
| 鼠标滚轮 | 缩放当前预览 |
| 左键拖动 | 平移放大的预览 |
| 双击预览 | 恢复适应窗口 |

## 项目结构

```text
Matching_Clothes/
├─ clothing_matcher_v8_ui_alt/  # 当前维护的 V8 对比界面版
│  ├─ app.py                    # Tkinter 桌面界面
│  ├─ engine.py                 # 匹配、OCR、颜色与导出逻辑
│  ├─ models/                   # 本地模型目录（模型不入 Git）
│  └─ verify_*.py               # 本地验证脚本
├─ clothing_matcher_exe/        # 历史版本，保留用于回溯
├─ clothing-stock-match/        # Codex 操作工作流
├─ examples/                    # 可公开试跑的合成图库、实拍图和预期结果
├─ scripts/                     # 模型下载等辅助脚本
├─ docs/                        # 使用、开发和模型说明
└─ AGENTS.md                    # Codex 项目级约束
```

## 数据与隐私

以下内容被 `.gitignore` 排除，不应提交到公开仓库：

- 原始款式图库和门店实拍图；
- 生成的标注结果、索引和本地学习记录；
- ONNX/PT/TorchScript 模型（兼容模型只作为独立 Release 资源发布，不进入 Git 历史）；
- `build/`、`dist/`、缓存和虚拟环境；
- 访问令牌、账号信息和其他密钥。

V8 的设置、索引、OCR 缓存、预览和本地反馈默认保存在 Windows 用户的 `%LOCALAPPDATA%\ClothingMatcherV8`。

## 开发与验证

开发环境、打包、验证命令和发布检查清单见 [开发与打包](docs/开发与打包.md)。核心验证命令：

```powershell
cd clothing_matcher_v8_ui_alt
python -m py_compile app.py engine.py verify_imports.py verify_export.py verify_v8.py
python verify_v8.py
python verify_imports.py
python verify_export.py
```

后面三个脚本会按需使用本机测试图库或样例图；缺少相应本地数据时，部分案例会跳过或无法执行。

## 准确率边界

- 图库中没有对应款时，应选择“未匹配”，程序不会凭空生成正确款。
- 大面积遮挡、极端裁剪或图库图本身被红字覆盖时，正确款不保证排第一。
- 模型输出只用于缩小人工检索范围，不应替代最终人工确认。
- 人工反馈仅在同一个图库路径下加权，更换图库后不会沿用旧图库结论。

## 参与项目

提交问题或改进前请阅读 [贡献指南](CONTRIBUTING.md)。报告安全或隐私问题请阅读 [安全说明](SECURITY.md)。

## 许可证

本项目目前尚未指定开源许可证。仓库公开表示代码可被查看，但不自动授予复制、修改或再分发权利；如需采用、商用或二次发布，请先联系仓库所有者。
