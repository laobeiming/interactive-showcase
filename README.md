# 多媒体沉浸式交互控制系统 (Interactive Showcase Prototype)

本项目为展厅多媒体交互装置原型设计方案，集成了高拍仪图像采集、二维码快速识别、纸张摆放姿态检测以及大屏多媒体联动控制。

---

## 🛠 技术栈
- **核心语言**：Python 3.10+
- **计算机视觉**：OpenCV (图像边缘计算、角度矫正、高拍仪推流)
- **解码组件**：PyZbar (二维码秒级解码匹配)
- **中文标注**：Pillow (OpenCV 预览画面渲染中文提示)
- **界面交互**：PyQt5 (多媒体全屏展示、无缝视频切换)
- **硬件集成**：USB 外部触发按键监听、Windows 开机自启服务

---

## 🚀 核心架构与功能流转
1. **待机常驻**：系统启动后全屏无缝循环播放待机视频；摄像头在后台持续扫描，不打断大屏画面。
2. **图纸扫描与姿态校准**：
   - 高拍仪实时捕捉观众手绘画作。
   - 算法识别图纸二维码 ID，并通过 `cv2.minAreaRect` 旋转外接矩形判断图纸水平摆放角度（超过阈值时给出摆正提示）。
3. **实体按键触发**：监听展台外部 USB 大按钮（映射为 `Space` / `Enter`），按下时校验「图纸已摆正 + 二维码有效」，通过后保存画作并播放快门反馈。
4. **场景联动**：根据二维码关联的画作主题，平滑切入对应的外星花园生物互动动效，居中展示观众抓拍画作。
5. **自动复位**：互动画面展示 8 秒后淡出，自动回到待机视频循环，等待下一位观众。

---

## ✅ 阶段一：基础骨架与待机展示（已完成）

### 当前文件结构
```
interactive-showcase/
├── main.py                 # 唯一完整入口：全屏展示 + 摄像头扫描 + 场景联动
├── scanner.py              # 摄像头采集、二维码识别与图纸姿态模块（可独立调试）
├── run.bat                 # Windows 双击启动（自动检查 / 安装依赖）
├── setup_autostart.bat     # 写入当前用户 Windows 启动文件夹，实现开机自启
├── requirements.txt        # Python 运行依赖
├── tests/                  # 自动化单元测试
├── assets/videos/          # 待机背景视频放置目录
└── captures/               # 现场调试快照与识别标注图（运行时自动生成）
```

### 1. 安装依赖
```powershell
cd "d:\多媒体沉浸式交互控制系统\interactive-showcase"
python -m pip install -r requirements.txt
```
- 也可直接双击 `run.bat`：脚本会自动定位 Python / `.venv`、检查并安装缺失依赖，然后全屏启动 `main.py`。

### 2. 启动待机展示
```powershell
python main.py                 # 全屏待机视频 + 后台摄像头扫描
python main.py --camera-preview  # 启动即显示摄像头预览（现场调纸用）
python main.py --windowed      # 1280x720 窗口运行，便于调试
python main.py --no-camera     # 只播放待机视频，不打开摄像头
python main.py --no-video      # 强制使用黑色呼吸文字待机页
```
- **Space / Enter**：校验通过后拍照并进入互动大屏；未摆正 / 未识别二维码时，画面中央弹出红色动态提示并拒绝触发。
- 按 **C** 切换摄像头预览 / 待机视频（摄像头仍在后台运行）；按 **ESC** 退出程序，按 **F11** 切换全屏 / 窗口。
- 将待机视频放入 `assets/videos/`（推荐命名 `standby.mp4`，H.264 编码）或项目根目录，程序会自动循环播放。
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
python main.py --windowed --no-camera       # 查看待机呼吸页
python scanner.py --demo                    # 模拟摄像头：动态画纸 + 二维码实时识别
python scanner.py --image demo/qr_demo.png  # 离线识别演示图片
python main.py --windowed --video demo/standby_demo.avi --no-camera  # 待机视频循环
```
- `--demo` 会生成“桌面上的手绘画纸 + 二维码”动态画面，**无需真实摄像头**；按 `q / ESC` 退出，按 `s` 截图。
- `demo/` 目录用于本地演示素材，已加入 `.gitignore`，不会推送到仓库。
- 没有真实摄像头时，仍然可以验证待机视频、呼吸画布与界面布局；拍照触发需在接入摄像头并识别到图纸后测试。

### 5. 说明
- 二维码识别依赖 `pyzbar`；Windows 安装包内已带 `libzbar` DLL，若提示加载失败，请安装 Microsoft Visual C++ 运行库。
- 待机视频默认静音，可用 `python main.py --volume 50` 调整。
- 现场调试快照目录 `captures/` 已加入 `.gitignore`，不会误提交。

---

## ✅ 阶段二：视觉检测与大屏联动（已完成）

### 1. 模块能力
- **scanner.py**：摄像头打开 / 读取 / 释放（含热插拔与读帧重试）、pyzbar 二维码识别、基于 `cv2.minAreaRect` 旋转外接矩形的图纸摆放角度检测。
  - `PaperPose`：输出中心点 `rect_center`、长宽 `rect_size`、标准化后的真实夹角 `angle_deg`（正数表示图纸右侧偏下），以及 `correction_hint`（“请逆时针 / 顺时针旋转图纸约 X°”）；
  - 预览画面用 `cv2.boxPoints()` + `cv2.drawContours()` 绘制贴合纸张倾斜的旋转线框：超过阈值（默认 ±5°）为红色并标注“请摆正图画（偏离 X°）”，阈值内为绿色并标注“已摆正”；
  - `ScanStatus`：`no_paper` / `tilted` / `no_code` / `ready`，只有 `ready` 允许上层触发联动；
  - `CameraScanner.analyze_frame()` 一帧同时返回二维码与图纸姿态；`analyze_frame(...).to_alignment()` 或 `detect_paper_alignment(frame)` 直接返回 `{is_straight, angle, code_id}`，主程序无需关心图像细节。
- **main.py**：`VisionWorker` 后台线程按约 8 FPS 分析画面，默认停留在待机视频；`Space / Enter` 按键触发后执行角度 + 二维码校验，拍照保存并进入主题互动页，8 秒后淡出回到待机视频。`C` 可切换摄像头预览，`tilted` 状态只提示校正、不会误触发大屏。

### 2. 启动联动
```powershell
python main.py                 # 全屏待机视频 + 后台摄像头扫描（默认行为）
python main.py --camera-preview  # 启动即显示摄像头预览，便于现场调纸
python main.py --windowed      # 窗口调试
python main.py --no-camera     # 禁用摄像头，仅待机展示
python main.py --tilt-threshold 5 --min-paper-ratio 0.04
python main.py --camera 1 --width 1920 --height 1080
```
- 默认显示待机视频；按 `C` 在摄像头预览 / 待机视频之间切换，摄像头保持后台运行。
- **Space / Enter** 对应外接 USB 大按钮：图纸已摆正且二维码有效时拍照进入互动页；否则中央弹出红色动态提示并拒绝触发。
- 按 `C` 在互动页立即结束展示并淡出；`--start-camera` 仍可传入，仅为兼容旧脚本保留。

**Windows 一键启动（推荐日常使用）**
- 双击 `run.bat`：自动定位 Python、检查依赖，随后全屏运行 `main.py`，默认播放待机视频并在后台打开摄像头扫描。
- 首次运行若缺少依赖，`run.bat` 会自动执行 `pip install -r requirements.txt`。
- 需要窗口调试时，可在命令行执行 `run.bat --windowed`；只演示待机页可执行 `run.bat --no-camera`。
- 若项目内存在 `.venv`，`run.bat` 会优先使用 `.venv\Scripts\python.exe`。

- 二维码内置主题关键字：`GARDEN / ALIEN`（外星花园）、`OCEAN / SEA / AQUA`（深海星潮）、`FOREST / TREE`（荧光森林）、`DESERT / SAND`（赤砂星球）；未知 ID 使用稳定兜底主题。

### 3. 终端自检与测试（无需界面）
```powershell
python scanner.py --selftest                            # 合成图纸：角度 + 二维码 + 联动状态
python main.py --no-camera --selftest --selftest-seconds 3   # 主程序无摄像头自检
python test_vision.py --demo --no-window --frames 30    # 无摄像头端到端联调
python test_vision.py --no-window --frames 60           # 真实摄像头逐帧联调
python -m unittest discover -s tests -v                 # 单元测试
```
- `test_vision.py` 会用 pyzbar 直接解码与 `scanner` 结果交叉校验；退出码 0 表示必需项全部通过。
- 真实摄像头联调时，若画面中没有图纸 / 二维码，识别项仅给出提示（INFO），仍会验证出帧与算法执行链路。

---

## ✅ 阶段三：按键触发 + 视频流转 + 开机自启（已完成）

### 1. 按键与拍照联动
- 键盘 `Space`、`Enter` / 小键盘 `Enter` 均映射为展台外接 USB 大按钮。
- 按键时读取最新一帧画面与识别结果，执行前置校验：
  - `is_straight == True` 且 `code_id` 非空：透视校正裁切画作 → 保存到 `captures/artwork_<时间>_<二维码ID>.jpg` → 播放合成快门声 + 全屏白闪 → 进入互动大屏。
  - 图纸未摆正：居中弹出红色动态提示（如“图纸未摆正，请调整后再按（偏离 15.0°）”），拒绝触发。
  - 未识别到二维码 / 扫描台未就绪：同样居中提示并拒绝触发。
- 外接 USB 大按钮通常以键盘设备形式接入，建议将其键值配置为 `Space` 或 `Enter`；可先在记事本中确认按键输出正常。

### 2. 大屏状态机
| 状态 | 画面 | 进入 / 退出 |
| --- | --- | --- |
| `standby` | 待机视频循环；无视频时使用内置星空呼吸画布 | 启动后默认进入；互动结束并淡出后回到此状态 |
| `interactive` | 主题互动动效 + 居中观众画作快照 | 按键校验通过后进入，默认展示 `8` 秒 |
| `reset` | 黑色淡出转场 | 展示结束自动进入，淡出完成后切回 `standby` |

- 待机视频播放失败或文件缺失时不会闪退，自动切到内置 `IdleCanvas` 呼吸画布。
- 需要调试时按 `C` 显示摄像头预览（可看到纸张角度与二维码框），再按 `C` 回到待机视频。
- 互动页按 `C` 可立即结束并淡出返回待机。
- `--scene-seconds 8` 可调整互动停留时长。

### 3. Windows 开机自启
```bat
setup_autostart.bat            :: 添加当前用户开机自启
setup_autostart.bat --check    :: 检查是否已启用
setup_autostart.bat --remove   :: 取消开机自启
```
- 脚本会在当前用户的 Windows 启动文件夹（`%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`）创建 `AlienGardenShowcase.lnk`，指向本项目 `run.bat`，并设置工作目录为项目根目录。
- 登录 Windows 后会自动启动全屏展示；建议展台电脑关闭睡眠、关闭屏保，并设置为自动登录。

### 4. 展厅部署步骤
1. 安装 Python 3.10+，执行 `python -m pip install -r requirements.txt`，或直接双击 `run.bat` 自动安装依赖。
2. 将待机视频放到 `assets/videos/standby.mp4`（H.264 / yuv420p / 1080p 推荐）。
3. 连接高拍仪与 USB 大按钮，执行 `python main.py --camera-preview` 调整纸张位置与拍摄范围。
4. 确认 `captures/` 可写，按键拍照后能生成 `artwork_*.jpg`。
5. 执行 `setup_autostart.bat`，重启电脑验证登录后是否自动全屏启动。
6. 定期清理 `captures/` 中的历史画作，避免磁盘占满。
