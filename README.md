<div align="center">

<a href="https://linux.do/" title="LINUX DO 社区">
  <img src="assets/linux-do.svg" width="88" height="88" alt="LINUX DO" />
</a>

### [LINUX&nbsp;DO](https://linux.do/)

**本项目在 [LINUX DO](https://linux.do/) 社区分享与交流** · 欢迎同好围观、反馈

[![LINUX DO](https://img.shields.io/badge/Community-LINUX%20DO-1c1c1e?style=for-the-badge&labelColor=ffb003&logoColor=white)](https://linux.do/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](./LICENSE)

</div>

---

# 一览清图 · RemovePicWatermark

为**重复出现的图片水印**建样式、批量找位置、再用 OpenCV / **LaMa** 修补；难图可手涂精修，可选 YOLO 训练增强检测。

> **免责声明**：请仅处理你有权处理的图片。使用后果由使用者自行承担。

| 项 | 内容 |
| --- | --- |
| 中文名 | **一览清图** |
| 包名 | `remove_pic_watermark` |
| 许可 | [MIT](./LICENSE) |
| 版本 | 见 `src/remove_pic_watermark/__init__.py` |

## 功能概要

| 模块 | 做什么 |
| --- | --- |
| **水印样式** | 样例框选/涂抹建档；可手改 mask / AI 抠图（BiRefNet） |
| **批量去除** | 多图队列；定位（固定/附近/全图）× 查找（样式 / 样式+模型 / 模型） |
| **单张精修** | 多图画布；涂抹/矩形框选 → 去除；格上恢复/去除/移除 |
| **训练检测** | 框选训 YOLO，写入 `watermark.pt`（可选） |
| **处理结果** | 查看历史任务 `workspace/jobs/` |

原理两步：**找水印（mask）→ 按 mask 修补**。

## 用到的模型

| 模型 | 作用 | 路径 |
| --- | --- | --- |
| **LaMa** (`big-lama.pt`) | 高质量修补 | `workspace/models/torch/hub/checkpoints/` |
| **BiRefNet** | 样式页 AI 抠模板 | `workspace/models/birefnet/` |
| **YOLOv8n / OBB** | 训练底座 | `workspace/models/yolo/` |
| **watermark.pt** | 水印检测（自训或附带） | `workspace/models/yolo/watermark.pt` |

模板匹配 / 残差扫描走 OpenCV，**不依赖**上述权重。  
开箱 zip 可内置权重；源码仓库默认**不**提交大权重文件。

**GPU**：NVIDIA + 较新驱动可加速 LaMa / 训练 / BiRefNet；无独显可 CPU。

## 环境

- Windows 10/11（GUI 与打包脚本按 Windows）
- Python **3.10+**（推荐 3.12）

## 从源码安装并运行（全功能）

下面在 **Windows + PowerShell** 下操作。会装上界面、YOLO 检测训练、样式 AI 抠图等依赖（体积较大，首次安装可能较久）。

```powershell
# 1. 下载源码
git clone https://github.com/DEKVIW/RemovePicWatermark.git
cd RemovePicWatermark

# 2. 创建并进入虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. 升级 pip，再安装本项目
python -m pip install -U pip
pip install -e ".[gui,yolo,matting]"

# 4. 启动图形界面
python .\run_gui.py
```

说明：

- `pip install -e ".[gui,yolo,matting]"`：可编辑安装，改源码后不用反复重装；`gui` 是界面，`yolo` 是检测/训练，`matting` 是样式页 BiRefNet 抠图。  
- 若只要界面、不要检测训练和 AI 抠图，可改成：`pip install -e ".[gui]"`。  
- 有 **NVIDIA 显卡** 时，建议另装带 CUDA 的 PyTorch（以 [pytorch.org](https://pytorch.org) 当前命令为准），再执行上面的 `pip install -e ...`，LaMa / 训练 / 抠图会更快。  
- 模型权重默认不在仓库里，请放到 `workspace/models/`（见上文「用到的模型」）。也可直接用 Releases 里的绿色包

软件内建议顺序：

1. **设置** → 修补方式、运行设备  
2. **水印样式** → 导入样例 → 框水印 → 保存  
3. **批量去除** → 加图 → 勾选样式 → 开始  
4. 难图用 **单张精修** → 涂抹/多框 → 去除 → 导出  

## 构建绿色包（Windows）

```powershell
# 先完成上面的虚拟环境与全功能安装，再打包
powershell -ExecutionPolicy Bypass -File scripts\build_gui_onedir.ps1 -SkipInstall
```

| 产物 | 说明 |
| --- | --- |
| `dist/RemovePicWatermark/` | 可运行目录 |
| `dist/releases/RemovePicWatermark-x.y.z.zip` | 分发压缩包 |

解压后运行 `RemovePicWatermark.exe` 或 `start.bat`（勿只拷贝 exe）。

## 项目结构

```text
.
├── src/remove_pic_watermark/   # 主代码
│   ├── detectors/              # 样式匹配 / 扫描 / YOLO
│   ├── backends/               # OpenCV / LaMa 修补
│   ├── services/               # 建档、批处理
│   ├── profiles/               # 样式档案
│   ├── gui/                    # PySide6 + Fluent
│   └── ...
├── packaging/                  # PyInstaller 入口与说明
├── scripts/                    # build_gui_onedir.ps1 等
├── tests/
├── configs/
├── assets/                     # 图标、社区 logo
├── pyproject.toml
├── LICENSE
└── README.md
```

**默认不进仓库**（见 `.gitignore`）：`.venv/`、`build/`、`dist/`、`docs/`、`workspace/`、权重 `*.pt` / `*.safetensors`、本地样例图与调试输出等。

## 技术栈

Python · OpenCV · NumPy · PySide6 · QFluentWidgets ·（可选）PyTorch / Ultralytics / Transformers · PyInstaller

## License

[MIT](./LICENSE)
