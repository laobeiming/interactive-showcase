# 多媒体沉浸式交互控制系统 (Interactive Showcase Prototype)

本项目为展厅多媒体交互装置原型设计方案，集成了高拍仪图像采集、二维码快速识别、纸张摆放姿态检测以及大屏多媒体联动控制。

---

## 🛠 技术栈
- **核心语言**：Python 3.10+
- **计算机视觉**：OpenCV (图像边缘计算、角度矫正、高拍仪推流)
- **解码组件**：PyZbar (二维码秒级解码匹配)
- **界面交互**：PyQt5 / PySide6 (多媒体全屏展示、无缝视频切换)
- **硬件集成**：USB 外部触发按键监听、Windows 开机自启服务

---

## 🚀 核心架构与功能流转
1. **待机常驻**：系统启动后全屏无缝循环播放待机背景视频。
2. **图纸扫描与姿态校准**：
   - 高拍仪实时捕捉观众手绘画作。
   - 算法识别图纸二维码 ID，并通过轮廓检测判断图纸水平摆放角度（误差超过阈值时给出摆正提示）。
3. **实体按键触发**：监听展台外部 USB 大按键信号，收到触发后将画作高清抓拍并投射至大屏。
4. **场景联动**：根据二维码关联的画作主题，平滑切入对应的外星生物交互特效视频。

---

## ✅ 阶段一：基础骨架与待机展示（已完成）

### 当前文件结构
```
interactive-showcase/
├── main.py                 # 全屏无边框待机展示窗口（视频循环 + 呼吸文字兜底）
├── scanner.py              # 摄像头采集与二维码识别模块（可独立预览）
├── requirements.txt        # Python 运行依赖
├── assets/videos/          # 待机背景视频放置目录
└── captures/               # 现场调试快照与识别标注图（运行时自动生成）
```

### 1. 安装依赖
```powershell
cd "d:\多媒体沉浸式交互控制系统\interactive-showcase"
python -m pip install -r requirements.txt
```

### 2. 启动待机展示
```powershell
python main.py                 # 全屏无边框运行
python main.py --windowed      # 1280x720 窗口运行，便于调试
python main.py --no-video      # 强制使用黑色呼吸文字待机页
```
- 按 **ESC** 退出程序，按 **F11** 切换全屏 / 窗口。
- 将待机视频放入 `assets/videos/`（推荐命名 `standby.mp4`，H.264 编码），程序会自动循环播放。
- 未找到视频或视频解码失败时，自动切换到 **纯黑背景 + 呼吸文字** 待机页，展厅不会黑屏。

### 3. 摄像头与二维码调试
```powershell
python scanner.py                        # 默认 0 号摄像头实时预览
python scanner.py --camera 1             # 指定摄像头
python scanner.py --list-cameras         # 探测可用摄像头
python scanner.py --image path/to/qr.png # 离线识别单张图片
```
- 预览窗口按 **q / ESC** 退出，按 **s** 保存快照到 `captures/`。
- 识别结果包含二维码内容、外接矩形与中心点，可直接供后续场景联动使用。

### 4. 没有摄像头 / 展台设备时如何演示
```powershell
python main.py --windowed                   # 查看待机呼吸页
python scanner.py --demo                    # 模拟摄像头：动态画纸 + 二维码实时识别
python scanner.py --image demo/qr_demo.png  # 离线识别演示图片
python main.py --windowed --video demo/standby_demo.avi  # 查看待机视频循环效果
```
- `--demo` 会生成“桌面上的手绘画纸 + 二维码”动态画面，**无需真实摄像头**；按 `q / ESC` 退出，按 `s` 截图。
- `demo/` 目录用于本地演示素材，已加入 `.gitignore`，不会推送到仓库。
- 当前阶段尚未接入实体按键与纸张姿态检测，硬件到货后可继续联调。

### 5. 说明
- 二维码识别依赖 `pyzbar`；Windows 安装包内已带 `libzbar` DLL，若提示加载失败，请安装 Microsoft Visual C++ 运行库。
- 待机视频默认静音，可用 `python main.py --volume 50` 调整。
- 现场调试快照目录 `captures/` 已加入 `.gitignore`，不会误提交。
