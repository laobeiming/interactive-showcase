"""外星花园交互展示程序 —— 唯一完整入口（全屏展示 + 摄像头识别联动）。

用法：
    python main.py                              # 全屏运行，启动即打开摄像头扫描
    python main.py --windowed                   # 1280x720 窗口调试
    python main.py --no-camera                  # 仅播放待机页，不打开摄像头
    python main.py --camera 0 --width 1920 --height 1080
    python main.py --windowed --no-video        # 无视频时使用呼吸文字待机页
    python main.py --no-camera --selftest       # 终端自检（不依赖摄像头）

默认行为：
    * 全屏无边框显示，启动后循环播放待机视频；
    * 摄像头在后台持续扫描，不打断待机画面；
    * 按下 Space / Enter（外接 USB 大按钮）时校验图纸角度与二维码，
      通过后保存画作并切入互动大屏，8 秒后淡出回到待机；
    * 没有待机视频时使用内置星空呼吸画布兜底。

快捷键：
    Space / Enter  拍照触发（对应展台外部 USB 大按钮）
    C              摄像头预览 / 待机视频切换（摄像头保持后台运行）
    ESC            退出程序
    F11            全屏 / 窗口切换（调试用）

状态机：
    STANDBY（待机视频） → 按键校验成功 → INTERACTIVE（互动动效）
    → 8 秒后 → RESET（淡出） → STANDBY

待机视频仅作为待机背景保留；没有视频时自动使用黑色呼吸文字待机页。
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import random
import re
import struct
import sys
import threading
import time
import wave
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except Exception:  # 摄像头模块缺失时仍允许待机页启动
    cv2 = None  # type: ignore[assignment]
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QFontMetrics,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
    QRadialGradient,
)
from PyQt5.QtWidgets import QApplication, QSizePolicy, QStackedLayout, QWidget

try:
    from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer, QMediaPlaylist
    from PyQt5.QtMultimediaWidgets import QVideoWidget

    MULTIMEDIA_AVAILABLE = True
    MULTIMEDIA_IMPORT_ERROR = ""
except Exception as exc:  # 缺少多媒体组件时不阻塞待机页启动
    MULTIMEDIA_AVAILABLE = False
    MULTIMEDIA_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# 视觉模块与界面解耦：导入失败时待机页仍可正常运行
try:
    from scanner import (
        DEFAULT_MIN_PAPER_AREA_RATIO,
        DEFAULT_TILT_THRESHOLD_DEG,
        CameraConfig,
        CameraScanner,
        FrameAnalysis,
        ScanStatus,
    )

    SCANNER_AVAILABLE = True
    SCANNER_IMPORT_ERROR = ""
except Exception as exc:  # OpenCV / pyzbar / numpy 任一缺失
    CameraConfig = None  # type: ignore[assignment]
    CameraScanner = None  # type: ignore[assignment]
    FrameAnalysis = None  # type: ignore[assignment]
    ScanStatus = None  # type: ignore[assignment]
    DEFAULT_MIN_PAPER_AREA_RATIO = 0.04
    DEFAULT_TILT_THRESHOLD_DEG = 5.0
    SCANNER_AVAILABLE = False
    SCANNER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "assets" / "videos"
CAPTURE_DIR = BASE_DIR / "captures"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".webm", ".m4v"}
PREFERRED_VIDEO_KEYWORDS = ("standby", "idle", "waiting", "daiji", "待机")

# 联动触发参数
TRIGGER_STABLE_FRAMES = 3
TRIGGER_COOLDOWN_SECONDS = 3.0
DEFAULT_SCENE_SECONDS = 8.0
CAMERA_ANALYSIS_INTERVAL = 0.12
CAMERA_FRAME_INTERVAL = 1.0 / 24.0

# 按键触发 / 状态机 / 转场参数
SCENE_FADE_OUT_MS = 480
SCENE_FADE_IN_MS = 520
SHUTTER_FLASH_MS = 180
PROMPT_DEFAULT_MS = 2400
BUTTON_DEBOUNCE_SECONDS = 0.45


class ShowcaseState(str, Enum):
    """大屏三态状态机：待机 / 互动 / 复位。"""

    STANDBY = "standby"
    INTERACTIVE = "interactive"
    RESET = "reset"

TITLE_FONT_CANDIDATES = (
    "HarmonyOS Sans SC",
    "Microsoft YaHei UI",
    "Microsoft YaHei",
    "Source Han Sans CN",
    "Noto Sans SC",
    "Noto Sans CJK SC",
    "SimHei",
)
LATIN_FONT_CANDIDATES = ("Segoe UI", "Helvetica Neue", "Arial")

LOGGER = logging.getLogger("alien_garden.main")


def pick_font_family(candidates: tuple[str, ...]) -> str:
    """在当前系统中挑选第一个可用字体，避免中文显示为方框。"""
    available = set(QFontDatabase().families())
    for family in candidates:
        if family in available:
            return family
    return QFont().defaultFamily()


def find_standby_video(explicit_path: Optional[Path] = None) -> Optional[Path]:
    """确定待机视频：优先显式路径，其次 assets/videos，最后项目根目录。"""
    if explicit_path is not None:
        path = explicit_path.expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        if path.is_file() and path.stat().st_size > 0:
            return path
        LOGGER.warning("指定的视频不可用：%s", path)
        return None

    candidates: list[Path] = []
    for directory in (VIDEO_DIR, BASE_DIR):
        if not directory.is_dir():
            continue
        for item in sorted(directory.iterdir()):
            if (
                item.is_file()
                and item.suffix.lower() in VIDEO_SUFFIXES
                and item.stat().st_size > 0
                and item not in candidates
            ):
                candidates.append(item)
    if not candidates:
        return None

    for item in candidates:
        stem = item.stem.lower()
        if any(keyword in stem for keyword in PREFERRED_VIDEO_KEYWORDS):
            return item
    return candidates[0]


@dataclass(frozen=True)
class SceneTheme:
    """二维码 ID 对应的场景联动主题。"""

    key: str
    title: str
    subtitle: str
    accent: Tuple[int, int, int]
    keywords: Tuple[str, ...] = ()


SCENE_THEMES = (
    SceneTheme(
        "garden",
        "外星花园",
        "第一株外星植物正在回应你的画笔",
        (96, 232, 184),
        ("GARDEN", "ALIEN", "外星", "花园", "001"),
    ),
    SceneTheme(
        "ocean",
        "深海星潮",
        "发光水母群在失重中缓慢游弋",
        (86, 188, 250),
        ("OCEAN", "SEA", "AQUA", "水母", "海洋"),
    ),
    SceneTheme(
        "forest",
        "荧光森林",
        "画笔唤醒了沉睡的孢子森林",
        (140, 226, 116),
        ("FOREST", "TREE", "森林", "孢子"),
    ),
    SceneTheme(
        "desert",
        "赤砂星球",
        "沙丘下的晶体开始共振",
        (240, 172, 96),
        ("DESERT", "SAND", "砂", "沙漠"),
    ),
)

CUSTOM_PALETTES = (
    (96, 232, 184),
    (120, 190, 255),
    (236, 160, 240),
    (255, 190, 110),
)


def resolve_scene_theme(qr_data: Optional[str]) -> SceneTheme:
    """把二维码内容映射为联动主题；未知 ID 使用稳定的随机配色兜底。"""
    text = (qr_data or "").strip()
    upper = text.upper()
    for theme in SCENE_THEMES:
        if any(keyword.upper() in upper for keyword in theme.keywords):
            return theme

    seed = sum(upper.encode("utf-8")) if upper else 0
    palette = CUSTOM_PALETTES[seed % len(CUSTOM_PALETTES)]
    suffix = text[-6:] if text else "未命名"
    return SceneTheme(
        "custom",
        "神秘星球",
        f"二维码「{suffix}」已接入外星生态",
        palette,
        (),
    )


def bgr_to_qimage(frame: Any) -> QImage:
    """把 OpenCV BGR 帧转换为独立持有内存的 QImage，避免悬垂指针。"""
    if frame is None:
        return QImage()
    if not isinstance(frame, np.ndarray) or frame.size == 0:
        return QImage()

    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)

    height, width = frame.shape[:2]
    data = frame.tobytes()
    if frame.ndim == 2:
        image = QImage(data, width, height, frame.strides[0], QImage.Format_Grayscale8)
    else:
        image = QImage(data, width, height, frame.strides[0], QImage.Format_BGR888)
    return image.copy()


@lru_cache(maxsize=1)
def build_shutter_wav() -> bytes:
    """合成一段短促快门“咔嗒”声，避免依赖外部音频素材。"""
    sample_rate = 44100
    duration = 0.12
    rng = random.Random(2024)
    frames = bytearray()
    total = int(sample_rate * duration)
    for index in range(total):
        t = index / sample_rate
        envelope = math.exp(-t * 46.0)
        click = math.sin(math.tau * 2100.0 * t) * 0.72
        noise = (rng.random() * 2.0 - 1.0) * 0.32
        second_click = 0.0
        if 0.032 <= t <= 0.052:
            second_click = math.sin(math.tau * 3200.0 * (t - 0.032)) * 0.55
        value = max(-1.0, min(1.0, (click + noise + second_click) * envelope))
        frames.extend(struct.pack("<h", int(value * 32767)))

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(bytes(frames))
    return buffer.getvalue()


class VisionWorker(QThread):
    """后台采集摄像头并分析二维码 / 图纸姿态，避免阻塞 Qt 主线程。"""

    frame_ready = pyqtSignal(object, object)
    camera_opened = pyqtSignal(str)
    camera_error = pyqtSignal(str)

    def __init__(
        self,
        config: Any,
        tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
        min_paper_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
        analysis_interval: float = CAMERA_ANALYSIS_INTERVAL,
        frame_interval: float = CAMERA_FRAME_INTERVAL,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._tilt_threshold = tilt_threshold
        self._min_paper_ratio = min_paper_ratio
        self._analysis_interval = max(0.05, float(analysis_interval))
        self._frame_interval = max(0.01, float(frame_interval))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """请求线程退出，并等待摄像头释放完成。"""
        self._stop_event.set()
        if self.isRunning():
            self.wait(3000)

    def run(self) -> None:  # noqa: D401 - QThread 入口
        if not SCANNER_AVAILABLE:
            self.camera_error.emit(f"视觉模块不可用：{SCANNER_IMPORT_ERROR}")
            return

        scanner = CameraScanner(self._config)
        if not scanner.open():
            self.camera_error.emit(scanner.last_error or "无法打开摄像头")
            return

        self.camera_opened.emit(
            f"摄像头 {self._config.index} 已连接 "
            f"({self._config.width}x{self._config.height} 请求分辨率)"
        )
        last_analysis = None
        last_analysis_at = 0.0
        read_failures = 0

        try:
            while not self._stop_event.is_set():
                loop_started = time.monotonic()
                frame = scanner.read_frame()
                if frame is None:
                    read_failures += 1
                    if read_failures >= 10:
                        self.camera_error.emit(scanner.last_error or "摄像头读取中断")
                        break
                    self._stop_event.wait(0.1)
                    continue
                read_failures = 0

                now = time.monotonic()
                if last_analysis is None or now - last_analysis_at >= self._analysis_interval:
                    try:
                        last_analysis = scanner.analyze_frame(
                            frame,
                            tilt_threshold=self._tilt_threshold,
                            min_paper_area_ratio=self._min_paper_ratio,
                        )
                    except Exception as exc:  # 单帧异常不应中断整个扫描线程
                        LOGGER.exception("视觉分析异常：%s", exc)
                    last_analysis_at = now

                if not self._stop_event.is_set():
                    self.frame_ready.emit(frame, last_analysis)

                delay = self._frame_interval - (time.monotonic() - loop_started)
                if delay > 0:
                    self._stop_event.wait(delay)
        finally:
            scanner.release()


class IdleCanvas(QWidget):
    """没有待机视频时的兜底画面：纯黑背景、光晕、轨道与呼吸文字。"""

    BREATH_PERIOD = 4.6  # 一次完整呼吸的秒数

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(800, 450)
        self._started_at = time.perf_counter()
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setInterval(33)  # 约 30 FPS
        self._refresh_timer.timeout.connect(self.update)
        self._cn_family = pick_font_family(TITLE_FONT_CANDIDATES)
        self._latin_family = pick_font_family(LATIN_FONT_CANDIDATES)

    # ---- 动画生命周期 -------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802 - 遵循 Qt 命名
        self._started_at = time.perf_counter()
        self._refresh_timer.start()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._refresh_timer.stop()
        super().hideEvent(event)

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started_at

    @property
    def breath(self) -> float:
        """0.0 ~ 1.0 的呼吸系数，周期为 BREATH_PERIOD 秒。"""
        phase = self.elapsed * (2.0 * math.pi / self.BREATH_PERIOD)
        return 0.5 + 0.5 * math.sin(phase - math.pi / 2.0)

    # ---- 绘制 ---------------------------------------------------------
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        width = max(1, self.width())
        height = max(1, self.height())
        breath = self.breath
        center = QPointF(width / 2.0, height * 0.43)

        painter.fillRect(self.rect(), Qt.black)
        self._paint_glow(painter, center, width, height, breath)
        self._paint_orbits(painter, center, width, height, breath)
        self._paint_corner_marks(painter, width, height, breath)
        self._paint_titles(painter, center, width, height, breath)
        self._paint_footer(painter, width, height)
        painter.end()

    def _paint_glow(self, painter, center, width, height, breath) -> None:
        gradient = QRadialGradient(center, max(width, height) * 0.72)
        gradient.setColorAt(0.0, QColor(24, 122, 96, int(20 + 30 * breath)))
        gradient.setColorAt(0.45, QColor(9, 62, 50, int(10 + 16 * breath)))
        gradient.setColorAt(1.0, QColor(0, 0, 0, 0))
        painter.fillRect(self.rect(), gradient)

    def _paint_orbits(self, painter, center, width, height, breath) -> None:
        radius = min(width * 0.46, height * 0.86)
        orbit = QRectF(
            center.x() - radius,
            center.y() - radius * 0.32,
            radius * 2,
            radius * 0.64,
        )
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(96, 226, 184, int(14 + 22 * breath)), 1.0))
        painter.drawEllipse(orbit)
        painter.setPen(QPen(QColor(96, 226, 184, int(8 + 14 * breath)), 1.0))
        painter.drawEllipse(
            orbit.adjusted(
                -radius * 0.08,
                -radius * 0.026,
                radius * 0.08,
                radius * 0.026,
            )
        )

        # 沿轨道缓慢环绕的光点，让静止画面保持“活着”的感觉
        angle = self.elapsed * 0.45
        dot = QPointF(
            center.x() + math.cos(angle) * radius,
            center.y() + math.sin(angle) * radius * 0.32,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(90, 235, 190, int(28 + 30 * breath)))
        painter.drawEllipse(dot, 9.0, 9.0)
        painter.setBrush(QColor(150, 255, 216, int(110 + 110 * breath)))
        painter.drawEllipse(dot, 2.6, 2.6)

    def _paint_corner_marks(self, painter, width, height, breath) -> None:
        margin = max(22.0, min(width, height) * 0.045)
        length = max(36.0, min(width, height) * 0.055)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(120, 236, 200, int(22 + 30 * breath)), 1.2))
        for left, top in ((True, True), (False, True), (True, False), (False, False)):
            x = margin if left else width - margin
            y = margin if top else height - margin
            dx = length if left else -length
            dy = length if top else -length
            painter.drawLine(QPointF(x, y), QPointF(x + dx, y))
            painter.drawLine(QPointF(x, y), QPointF(x, y + dy))

    def _paint_titles(self, painter, center, width, height, breath) -> None:
        base = min(width, height)
        title_size = int(max(46, min(168, base * 0.118)))
        kicker_size = int(max(11, min(24, base * 0.022)))
        status_size = int(max(13, min(27, base * 0.026)))
        title_rect = QRectF(
            0,
            center.y() - title_size * 0.60,
            width,
            title_size * 1.30,
        )

        # 英文引导字
        kicker_font = QFont(self._latin_family)
        kicker_font.setPixelSize(kicker_size)
        kicker_font.setWeight(QFont.Medium)
        kicker_font.setLetterSpacing(QFont.AbsoluteSpacing, kicker_size * 0.45)
        painter.setFont(kicker_font)
        painter.setPen(QColor(132, 214, 190, int(70 + 80 * breath)))
        painter.drawText(
            QRectF(0, title_rect.top() - kicker_size * 2.4, width, kicker_size * 2),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "ALIEN GARDEN",
        )

        # 主标题：先画一圈低透明度光晕，再叠加呼吸中的实体字
        title_font = QFont(self._cn_family)
        title_font.setPixelSize(title_size)
        title_font.setWeight(QFont.DemiBold)
        title_font.setLetterSpacing(QFont.PercentageSpacing, 118)
        painter.setFont(title_font)
        painter.setPen(QColor(44, 255, 190, int(16 + 26 * breath)))
        for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3)):
            painter.drawText(
                title_rect.translated(dx, dy),
                Qt.AlignHCenter | Qt.AlignVCenter,
                "外星花园",
            )
        painter.setPen(QColor(228, 248, 240, int(175 + 75 * breath)))
        painter.drawText(title_rect, Qt.AlignHCenter | Qt.AlignVCenter, "外星花园")

        # 标题下方的呼吸分隔线
        line_y = title_rect.bottom() + base * 0.038
        line_width = min(width * 0.26, 460.0)
        gradient = QLinearGradient(
            width / 2.0 - line_width / 2.0,
            0,
            width / 2.0 + line_width / 2.0,
            0,
        )
        gradient.setColorAt(0.0, QColor(92, 236, 196, 0))
        gradient.setColorAt(0.5, QColor(92, 236, 196, int(55 + 70 * breath)))
        gradient.setColorAt(1.0, QColor(92, 236, 196, 0))
        painter.setPen(QPen(QBrush(gradient), 1.2))
        painter.drawLine(
            QPointF(width / 2.0 - line_width / 2.0, line_y),
            QPointF(width / 2.0 + line_width / 2.0, line_y),
        )

        # 待机状态文字
        status_font = QFont(self._cn_family)
        status_font.setPixelSize(status_size)
        status_font.setLetterSpacing(QFont.PercentageSpacing, 135)
        painter.setFont(status_font)
        painter.setPen(QColor(148, 220, 200, int(80 + 80 * breath)))
        painter.drawText(
            QRectF(0, line_y + base * 0.030, width, status_size * 2),
            Qt.AlignHCenter | Qt.AlignTop,
            "待 机 中   ·   S T A N D B Y",
        )

    def _paint_footer(self, painter, width, height) -> None:
        base = min(width, height)
        hint_size = int(max(10, min(17, base * 0.016)))
        hint_font = QFont(self._cn_family)
        hint_font.setPixelSize(hint_size)
        hint_font.setLetterSpacing(QFont.PercentageSpacing, 108)
        painter.setFont(hint_font)
        painter.setPen(QColor(180, 210, 200, 60))
        painter.drawText(
            QRectF(0, height - height * 0.13, width, hint_size * 2),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "请将手绘画作放置于扫描区，然后按下展台按键",
        )
        painter.setPen(QColor(150, 180, 170, 55))
        painter.drawText(
            QRectF(max(0.0, width - 280.0), height - 76.0, 220.0, 44.0),
            Qt.AlignRight | Qt.AlignVCenter,
            "ESC 退出",
        )


class CameraCanvas(QWidget):
    """实时扫描画面：摄像头预览 + 图纸轮廓 + 倾斜校正提示 + 识别结果。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(800, 450)

        self._image = QImage()
        self._analysis: Any = None
        self._error = ""
        self._frames = 0
        self._fps = 0.0
        self._last_frame_at = 0.0
        self._pulse = 0

        self._pulse_timer = QTimer(self)
        self._pulse_timer.setInterval(90)
        self._pulse_timer.timeout.connect(self._on_pulse)
        self._pulse_timer.start()

        family = pick_font_family(TITLE_FONT_CANDIDATES)
        latin = pick_font_family(LATIN_FONT_CANDIDATES)
        self._title_font = QFont(family, 21, QFont.DemiBold)
        self._status_font = QFont(family, 18, QFont.DemiBold)
        self._body_font = QFont(family, 12)
        self._small_font = QFont(family, 10)
        self._angle_font = QFont(latin, 26, QFont.DemiBold)
        self._mono_font = QFont(latin, 10)

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def last_analysis(self) -> Any:
        return self._analysis

    def set_frame(self, frame: Any, analysis: Any) -> None:
        self._image = bgr_to_qimage(frame)
        self._analysis = analysis
        self._error = ""

        now = time.monotonic()
        if self._last_frame_at > 0.0:
            instant = 1.0 / max(1e-3, now - self._last_frame_at)
            self._fps = instant if self._fps <= 0.0 else self._fps * 0.85 + instant * 0.15
        self._last_frame_at = now
        self._frames += 1
        self.update()

    def set_error(self, message: str) -> None:
        self._error = message
        LOGGER.warning("摄像头状态：%s", message)
        self.update()

    def clear(self) -> None:
        self._image = QImage()
        self._analysis = None
        self._last_frame_at = 0.0
        self._fps = 0.0
        self.update()

    def _on_pulse(self) -> None:
        self._pulse = (self._pulse + 1) % 1000
        self.update()

    # ---- 绘制 ---------------------------------------------------------
    def _status_style(self) -> Tuple[QColor, str, str]:
        analysis = self._analysis
        if analysis is None:
            return QColor(150, 178, 184), "正在连接扫描台", "请将画作放入高拍仪扫描区"
        value = getattr(getattr(analysis, "status", None), "value", "")
        message = getattr(analysis, "message", "")
        if value == "ready":
            return QColor(96, 232, 184), "可以拍照", "图纸已摆正 · 二维码有效 · 请按 Space / Enter"
        if value == "tilted":
            return QColor(255, 184, 92), "请摆正图纸", message
        if value == "no_code":
            return QColor(120, 196, 255), "等待二维码", message
        return QColor(205, 212, 214), "未检测到图纸", message

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        bounds = QRectF(self.rect())
        painter.fillRect(bounds, QColor(6, 10, 14))

        target: Optional[QRectF] = None
        if not self._image.isNull():
            scale = min(
                bounds.width() / self._image.width(),
                bounds.height() / self._image.height(),
            )
            draw_width = self._image.width() * scale
            draw_height = self._image.height() * scale
            target = QRectF(
                bounds.center().x() - draw_width / 2.0,
                bounds.center().y() - draw_height / 2.0,
                draw_width,
                draw_height,
            )
            painter.drawImage(target, self._image)
            self._draw_paper_overlay(painter, target)
            self._draw_vignette(painter, bounds)
        else:
            self._draw_idle(painter, bounds)

        self._draw_hud(painter, bounds, target)
        painter.end()

    def _draw_idle(self, painter: QPainter, bounds: QRectF) -> None:
        glow = QRadialGradient(bounds.center(), max(bounds.width(), bounds.height()) * 0.55)
        glow.setColorAt(0.0, QColor(24, 74, 66, 150))
        glow.setColorAt(1.0, QColor(4, 8, 12, 0))
        painter.fillRect(bounds, QBrush(glow))

        center = bounds.center()
        pulse = (math.sin(self._pulse * 0.18) + 1.0) / 2.0
        radius = min(bounds.width(), bounds.height()) * (0.10 + 0.02 * pulse)
        painter.setPen(QPen(QColor(90, 220, 180, 90), 2.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, radius, radius)
        painter.setPen(QPen(QColor(90, 220, 180, 30), 1.0))
        painter.drawEllipse(center, radius * 1.45, radius * 1.45)

        painter.setFont(self._status_font)
        painter.setPen(QColor(196, 236, 224))
        painter.drawText(
            QRectF(bounds.left(), center.y() + radius * 0.35, bounds.width(), 44),
            Qt.AlignHCenter | Qt.AlignTop,
            "等待摄像头画面",
        )
        painter.setFont(self._body_font)
        painter.setPen(QColor(144, 176, 170))
        detail = self._error or "正在初始化高拍仪，请稍候…"
        painter.drawText(
            QRectF(bounds.left() + 40, center.y() + radius * 0.35 + 44, bounds.width() - 80, 50),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            detail,
        )

        pen = QPen(QColor(96, 232, 184, 70), 1.4, Qt.DashLine)
        pen.setDashPattern([6, 6])
        painter.setPen(pen)
        frame_rect = QRectF(
            center.x() - bounds.width() * 0.19,
            center.y() - bounds.height() * 0.20,
            bounds.width() * 0.38,
            bounds.height() * 0.40,
        )
        painter.drawRoundedRect(frame_rect, 18, 18)


    def _draw_paper_overlay(self, painter: QPainter, target: QRectF) -> None:
        analysis = self._analysis
        if analysis is None:
            return
        pose = getattr(analysis, "pose", None)
        if pose is None or not getattr(pose, "found", False):
            pen = QPen(QColor(120, 210, 190, 65), 1.2, Qt.DashLine)
            pen.setDashPattern([5, 7])
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(
                QRectF(
                    target.left() + target.width() * 0.18,
                    target.top() + target.height() * 0.16,
                    target.width() * 0.64,
                    target.height() * 0.68,
                ),
                24,
                24,
            )
            return

        frame_size = getattr(analysis, "frame_size", (0, 0))
        if not frame_size or frame_size[0] <= 0 or frame_size[1] <= 0:
            return
        scale_x = target.width() / frame_size[0]
        scale_y = target.height() / frame_size[1]

        accent = QColor(96, 232, 184) if not pose.needs_correction else QColor(255, 92, 92)
        polygon = QPolygonF(
            [
                QPointF(target.left() + point[0] * scale_x, target.top() + point[1] * scale_y)
                for point in pose.quad
            ]
        )
        fill = QColor(accent)
        fill.setAlpha(28)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(accent, 2.6))
        painter.drawPolygon(polygon)

        points = list(polygon)
        for index in range(4):
            start = points[index]
            end = points[(index + 1) % 4]
            painter.setPen(QPen(accent, 5.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(start, end)
            painter.setPen(QPen(QColor(6, 12, 14), 2.0, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(start, end)

        center = QPointF(
            target.left() + pose.center[0] * scale_x,
            target.top() + pose.center[1] * scale_y,
        )
        painter.setPen(QPen(accent, 2.0))
        painter.drawLine(QPointF(center.x() - 14, center.y()), QPointF(center.x() + 14, center.y()))
        painter.drawLine(QPointF(center.x(), center.y() - 14), QPointF(center.x(), center.y() + 14))


    def _draw_vignette(self, painter: QPainter, bounds: QRectF) -> None:
        top = QLinearGradient(0, 0, 0, 170)
        top.setColorAt(0.0, QColor(4, 8, 12, 225))
        top.setColorAt(1.0, QColor(4, 8, 12, 0))
        painter.fillRect(QRectF(0, 0, bounds.width(), 170), QBrush(top))

        bottom = QLinearGradient(0, bounds.height() - 190, 0, bounds.height())
        bottom.setColorAt(0.0, QColor(4, 8, 12, 0))
        bottom.setColorAt(1.0, QColor(4, 8, 12, 235))
        painter.fillRect(QRectF(0, bounds.height() - 190, bounds.width(), 190), QBrush(bottom))


    def _draw_hud(self, painter: QPainter, bounds: QRectF, target: Optional[QRectF]) -> None:
        accent, status_title, status_message = self._status_style()
        analysis = self._analysis

        # 顶部左侧标题
        painter.setFont(self._mono_font)
        painter.setPen(QColor(96, 232, 184, 205))
        painter.drawText(
            QRectF(38, 20, 520, 18),
            Qt.AlignLeft | Qt.AlignVCenter,
            "ALIEN GARDEN  /  SCAN STATION",
        )
        painter.setFont(self._title_font)
        painter.setPen(QColor(236, 248, 244))
        painter.drawText(
            QRectF(36, 38, 520, 34),
            Qt.AlignLeft | Qt.AlignVCenter,
            "高拍仪实时扫描",
        )
        qr_text = getattr(analysis, "code_data", None) if analysis is not None else None
        subtitle = f"QR  {qr_text}" if qr_text else "图纸水平校准 · 二维码识别 · 姿态通过后自动联动"
        painter.setFont(self._small_font)
        painter.setPen(QColor(152, 182, 178))
        painter.drawText(
            QRectF(38, 74, max(240.0, bounds.width() * 0.62), 20),
            Qt.AlignLeft | Qt.AlignVCenter,
            subtitle,
        )

        # 右上角 LIVE 指示
        live = target is not None and not self._error
        pill_rect = QRectF(bounds.width() - 204, 30, 168, 34)
        pill_bg = QColor(8, 24, 20, 205) if live else QColor(30, 20, 12, 205)
        pill_accent = QColor(96, 232, 184) if live else QColor(255, 184, 92)
        painter.setBrush(QBrush(pill_bg))
        painter.setPen(QPen(QColor(pill_accent.red(), pill_accent.green(), pill_accent.blue(), 130), 1.2))
        painter.drawRoundedRect(pill_rect, 17, 17)

        pulse = (math.sin(self._pulse * 0.22) + 1.0) / 2.0
        dot_color = QColor(pill_accent)
        dot_color.setAlpha(int(140 + 100 * pulse))
        painter.setBrush(QBrush(dot_color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPointF(pill_rect.left() + 18, pill_rect.center().y()), 5.0, 5.0)

        live_text = f"LIVE  {self._fps:0.0f} FPS" if live else "OFFLINE"
        painter.setFont(self._mono_font)
        painter.setPen(QColor(224, 244, 238) if live else QColor(255, 210, 170))
        painter.drawText(
            QRectF(pill_rect.left() + 32, pill_rect.top(), pill_rect.width() - 40, pill_rect.height()),
            Qt.AlignLeft | Qt.AlignVCenter,
            live_text,
        )


        # 底部状态面板
        panel = QRectF(32, bounds.height() - 142, bounds.width() - 64, 104)
        painter.setBrush(QBrush(QColor(4, 10, 12, 216)))
        border = QColor(accent)
        border.setAlpha(120)
        painter.setPen(QPen(border, 1.3))
        painter.drawRoundedRect(panel, 18, 18)

        dot_center = QPointF(panel.left() + 32, panel.top() + 34)
        glow = QRadialGradient(dot_center, 26)
        glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 150))
        glow.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(dot_center, 26, 26)
        painter.setBrush(QBrush(accent))
        painter.drawEllipse(dot_center, 6.0, 6.0)

        painter.setFont(self._status_font)
        painter.setPen(QColor(238, 250, 246))
        painter.drawText(
            QRectF(panel.left() + 52, panel.top() + 16, panel.width() * 0.52, 32),
            Qt.AlignLeft | Qt.AlignVCenter,
            status_title,
        )
        painter.setFont(self._body_font)
        painter.setPen(QColor(162, 190, 186))
        message_width = max(160.0, panel.width() * 0.52)
        elided = QFontMetrics(self._body_font).elidedText(status_message, Qt.ElideRight, int(message_width))
        painter.drawText(
            QRectF(panel.left() + 52, panel.top() + 54, message_width, 28),
            Qt.AlignLeft | Qt.AlignVCenter,
            elided,
        )


        # 右侧倾斜角
        pose = getattr(analysis, "pose", None) if analysis is not None else None
        angle_rect = QRectF(panel.right() - 210, panel.top() + 12, 190, 78)
        painter.setFont(self._small_font)
        painter.setPen(QColor(140, 172, 168))
        painter.drawText(
            QRectF(angle_rect.left(), angle_rect.top(), angle_rect.width(), 18),
            Qt.AlignRight | Qt.AlignVCenter,
            "PAPER TILT",
        )
        if pose is not None and getattr(pose, "found", False):
            angle_color = QColor(96, 232, 184) if not pose.needs_correction else QColor(255, 92, 92)
            painter.setFont(self._angle_font)
            painter.setPen(angle_color)
            painter.drawText(
                QRectF(angle_rect.left(), angle_rect.top() + 14, angle_rect.width(), 44),
                Qt.AlignRight | Qt.AlignVCenter,
                f"{pose.angle_deg:+.1f}°",
            )
            hint = (
                getattr(pose, "correction_hint", "")
                if pose.needs_correction
                else f"在阈值 ±{pose.tilt_threshold:.1f}° 内"
            )
            painter.setFont(self._small_font)
            painter.setPen(angle_color if pose.needs_correction else QColor(140, 172, 168))
            painter.drawText(
                QRectF(angle_rect.left(), angle_rect.top() + 56, angle_rect.width(), 20),
                Qt.AlignRight | Qt.AlignVCenter,
                hint,
            )
        else:
            painter.setFont(self._body_font)
            painter.setPen(QColor(140, 172, 168))
            painter.drawText(
                QRectF(angle_rect.left(), angle_rect.top() + 20, angle_rect.width(), 44),
                Qt.AlignRight | Qt.AlignVCenter,
                "--",
            )

        painter.setFont(self._small_font)
        painter.setPen(QColor(128, 158, 154))
        painter.drawText(
            QRectF(32, bounds.height() - 32, bounds.width() - 64, 20),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "C 预览切换   ·   Space / Enter 拍照   ·   ESC 退出",
        )


class SceneCanvas(QWidget):
    """场景联动页：二维码匹配成功后，大屏切入对应主题的沉浸式画面。"""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent, True)
        self.setAutoFillBackground(False)
        self.setMinimumSize(800, 450)

        self._theme: Optional[SceneTheme] = None
        self._qr_data = ""
        self._snapshot = QImage()
        self._started_at = 0.0
        self._duration = DEFAULT_SCENE_SECONDS
        self._stars: list[Tuple[float, float, float, float, float]] = []

        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self.update)

        self._family = pick_font_family(TITLE_FONT_CANDIDATES)
        self._latin = pick_font_family(LATIN_FONT_CANDIDATES)
        self._kicker_font = QFont(self._latin, 11, QFont.DemiBold)
        self._body_font = QFont(self._family, 14)
        self._small_font = QFont(self._family, 11)
        self._mono_font = QFont(self._latin, 10)

    @property
    def theme(self) -> Optional[SceneTheme]:
        return self._theme

    def set_scene(
        self,
        theme: SceneTheme,
        qr_data: str,
        snapshot: Optional[QImage] = None,
        duration: float = DEFAULT_SCENE_SECONDS,
    ) -> None:
        self._theme = theme
        self._qr_data = qr_data
        self._snapshot = snapshot if snapshot is not None else QImage()
        self._duration = max(1.0, float(duration))
        self._started_at = time.monotonic()

        seed = sum((qr_data or theme.key).encode("utf-8"))
        rng = random.Random(seed)
        self._stars = [
            (
                rng.random(),
                rng.random(),
                rng.uniform(0.8, 2.4),
                rng.uniform(0.0, math.tau),
                rng.uniform(0.35, 1.15),
            )
            for _ in range(72)
        ]
        self._timer.start()
        self.update()

    def clear_scene(self) -> None:
        self._timer.stop()
        self._snapshot = QImage()
        self.update()

    def remaining_seconds(self) -> float:
        if self._started_at <= 0.0:
            return 0.0
        return max(0.0, self._duration - (time.monotonic() - self._started_at))

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        bounds = QRectF(self.rect())
        theme = self._theme or resolve_scene_theme("")
        accent = QColor(*theme.accent)
        elapsed = max(0.0, time.monotonic() - self._started_at)

        self._draw_background(painter, bounds, accent)
        self._draw_stars(painter, bounds, accent, elapsed)
        self._draw_text_block(painter, bounds, theme, accent)
        self._draw_snapshot_card(painter, bounds, theme, accent)
        self._draw_progress(painter, bounds, accent)
        painter.end()

    def _draw_background(self, painter: QPainter, bounds: QRectF, accent: QColor) -> None:
        background = QLinearGradient(0, 0, bounds.width(), bounds.height())
        background.setColorAt(0.0, QColor(4, 9, 12))
        background.setColorAt(0.55, accent.darker(520))
        background.setColorAt(1.0, QColor(2, 5, 8))
        painter.fillRect(bounds, QBrush(background))

        glow = QRadialGradient(
            QPointF(bounds.width() * 0.34, bounds.height() * 0.48),
            max(bounds.width(), bounds.height()) * 0.62,
        )
        glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 95))
        glow.setColorAt(0.45, QColor(accent.red(), accent.green(), accent.blue(), 28))
        glow.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        painter.fillRect(bounds, QBrush(glow))

        side_glow = QRadialGradient(
            QPointF(bounds.width() * 0.82, bounds.height() * 0.18),
            bounds.width() * 0.45,
        )
        side_glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 52))
        side_glow.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        painter.fillRect(bounds, QBrush(side_glow))

    def _draw_stars(
        self,
        painter: QPainter,
        bounds: QRectF,
        accent: QColor,
        elapsed: float,
    ) -> None:
        painter.setPen(Qt.NoPen)
        for x_ratio, y_ratio, radius, phase, speed in self._stars:
            x = (x_ratio * bounds.width() + math.sin(elapsed * 0.22 * speed + phase) * 12) % bounds.width()
            y = (y_ratio * bounds.height() + math.cos(elapsed * 0.16 * speed + phase) * 8) % bounds.height()
            twinkle = (math.sin(elapsed * 1.5 * speed + phase) + 1.0) / 2.0
            color = QColor(230, 255, 246, int(35 + 150 * twinkle))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(QPointF(x, y), radius, radius)


    def _draw_text_block(
        self,
        painter: QPainter,
        bounds: QRectF,
        theme: SceneTheme,
        accent: QColor,
    ) -> None:
        block_width = min(bounds.width() * 0.72, 980.0)
        left = bounds.center().x() - block_width / 2.0

        painter.setFont(self._kicker_font)
        painter.setPen(accent)
        painter.drawText(
            QRectF(left, bounds.height() * 0.045, block_width, 22),
            Qt.AlignHCenter | Qt.AlignVCenter,
            "SCENE LINKED  /  二维码已匹配",
        )

        title_size = int(max(26, min(64, bounds.height() * 0.072)))
        title_font = QFont(self._family, title_size, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(240, 252, 248))
        painter.drawText(
            QRectF(left, bounds.height() * 0.075, block_width, title_size * 1.2),
            Qt.AlignHCenter | Qt.AlignVCenter,
            theme.title,
        )

        painter.setFont(self._body_font)
        painter.setPen(QColor(176, 206, 198))
        painter.drawText(
            QRectF(
                left,
                bounds.height() * 0.075 + title_size * 1.25,
                block_width,
                44,
            ),
            Qt.AlignHCenter | Qt.AlignTop | Qt.TextWordWrap,
            theme.subtitle,
        )


    def _draw_snapshot_card(
        self,
        painter: QPainter,
        bounds: QRectF,
        theme: SceneTheme,
        accent: QColor,
    ) -> None:
        card_height = min(bounds.height() * 0.60, bounds.width() * 0.44 * 1.18)
        card_width = card_height / 1.18
        card = QRectF(
            bounds.center().x() - card_width / 2.0,
            bounds.height() * 0.30,
            card_width,
            card_height,
        )

        glow = QRadialGradient(card.center(), card_width * 0.95)
        glow.setColorAt(0.0, QColor(accent.red(), accent.green(), accent.blue(), 85))
        glow.setColorAt(1.0, QColor(accent.red(), accent.green(), accent.blue(), 0))
        painter.fillRect(card.adjusted(-60, -60, 60, 60), QBrush(glow))

        painter.setBrush(QBrush(QColor(6, 16, 18, 214)))
        painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 130), 1.4))
        painter.drawRoundedRect(card, 22, 22)

        painter.setFont(self._small_font)
        painter.setPen(QColor(150, 184, 176))
        painter.drawText(
            QRectF(card.left() + 20, card.top() + 14, card.width() - 40, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "观众作品  /  ARTWORK SNAPSHOT",
        )

        image_area = QRectF(
            card.left() + 16,
            card.top() + 44,
            card.width() - 32,
            card.height() - 118,
        )
        clip = QPainterPath()
        clip.addRoundedRect(image_area, 14, 14)
        painter.save()
        painter.setClipPath(clip)
        painter.fillRect(image_area, QColor(3, 8, 10))

        if not self._snapshot.isNull():
            scale = min(
                image_area.width() / self._snapshot.width(),
                image_area.height() / self._snapshot.height(),
            )
            draw_width = self._snapshot.width() * scale
            draw_height = self._snapshot.height() * scale
            target = QRectF(
                image_area.center().x() - draw_width / 2.0,
                image_area.center().y() - draw_height / 2.0,
                draw_width,
                draw_height,
            )
            painter.drawImage(target, self._snapshot)
        else:
            painter.setFont(self._body_font)
            painter.setPen(QColor(120, 160, 154))
            painter.drawText(image_area, Qt.AlignCenter, "作品快照")
        painter.restore()

        painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 90), 1.0))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(image_area, 14, 14)

        painter.setFont(self._small_font)
        painter.setPen(accent)
        qr_label = self._qr_data or theme.key.upper()
        qr_label = QFontMetrics(self._small_font).elidedText(
            qr_label, Qt.ElideMiddle, int(card.width() - 40)
        )
        painter.drawText(
            QRectF(card.left() + 20, card.bottom() - 58, card.width() - 40, 40),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            f"{theme.title} · 已投射至大屏\nID {qr_label}",
        )

    def _draw_progress(self, painter: QPainter, bounds: QRectF, accent: QColor) -> None:
        painter.setFont(self._mono_font)
        painter.setPen(QColor(120, 156, 150))
        painter.drawText(
            QRectF(bounds.width() * 0.075, 24, bounds.width() * 0.40, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "ALIEN GARDEN  /  IMMERSIVE SHOWCASE",
        )
        painter.drawText(
            QRectF(bounds.width() * 0.55, 24, bounds.width() * 0.375, 22),
            Qt.AlignRight | Qt.AlignVCenter,
            f"C 返回扫描台  ·  {self.remaining_seconds():.0f} 秒后自动返回",
        )

        track = QRectF(
            bounds.width() * 0.075,
            bounds.height() - 44,
            bounds.width() * 0.85,
            4,
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 255, 255, 22)))
        painter.drawRoundedRect(track, 2, 2)
        ratio = (self.remaining_seconds() / self._duration) if self._duration > 0 else 0.0
        fill = QRectF(track.left(), track.top(), track.width() * max(0.0, min(1.0, ratio)), track.height())
        painter.setBrush(QBrush(accent))
        painter.drawRoundedRect(fill, 2, 2)


class ShowcaseOverlay(QWidget):
    """覆盖在大屏最上层的黑色淡出层 + 居中动态提示。

    使用独立的 ``Qt.Tool`` 顶层窗口，确保 QVideoWidget 播放待机视频时，
    提示文字和淡入淡出也能稳定覆盖在画面上。
    """

    def __init__(self, anchor: QWidget) -> None:
        super().__init__(
            anchor,
            Qt.Tool
            | Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setFocusPolicy(Qt.NoFocus)

        self._fade_color = QColor(0, 0, 0)
        self._fade_opacity = 0.0
        self._fade_from = 0.0
        self._fade_target = 0.0
        self._fade_started = 0.0
        self._fade_duration = 1
        self._fade_running = False
        self._fade_callback: Any = None

        self._message = ""
        self._subtitle = ""
        self._message_started = 0.0
        self._message_visible = False
        self._accent = QColor(255, 92, 92)

        family = pick_font_family(TITLE_FONT_CANDIDATES)
        self._title_font = QFont(family, 27, QFont.DemiBold)
        self._subtitle_font = QFont(family, 14)
        self._icon_font = QFont(family, 25, QFont.Bold)

        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(16)
        self._anim_timer.timeout.connect(self._on_tick)

        self._message_timer = QTimer(self)
        self._message_timer.setSingleShot(True)
        self._message_timer.timeout.connect(self.clear_message)

        self.hide()

    # ---- 生命周期 / 几何同步 ------------------------------------------
    def sync_geometry(self, anchor: QWidget) -> None:
        """让覆盖层始终与主窗口客户区重合。"""
        self.setGeometry(anchor.geometry())
        if self.isVisible():
            self.raise_()

    def _ensure_visible(self) -> None:
        if self.width() <= 0 or self.height() <= 0:
            return
        if not self.isVisible():
            self.show()
        self.raise_()

    def _maybe_hide(self) -> None:
        if not self._message_visible and not self._fade_running:
            self._anim_timer.stop()
            self.hide()

    # ---- 淡入淡出 -----------------------------------------------------
    def fade_to(
        self,
        target: float,
        duration_ms: int = 400,
        *,
        color: Tuple[int, int, int] = (0, 0, 0),
        on_finished: Any = None,
        start_opacity: Optional[float] = None,
    ) -> None:
        if start_opacity is not None:
            self._fade_opacity = max(0.0, min(1.0, float(start_opacity)))
        self._fade_from = self._fade_opacity
        self._fade_target = max(0.0, min(1.0, float(target)))
        self._fade_color = QColor(*color)
        self._fade_duration = max(1, int(duration_ms))
        self._fade_started = time.monotonic()
        self._fade_running = True
        self._fade_callback = on_finished
        self._ensure_visible()
        self._anim_timer.start()
        self.update()

    def flash(
        self,
        color: Tuple[int, int, int] = (255, 255, 255),
        duration_ms: int = SHUTTER_FLASH_MS,
        *,
        on_finished: Any = None,
    ) -> None:
        """模拟快门白闪。"""
        self.fade_to(
            0.0,
            duration_ms,
            color=color,
            on_finished=on_finished,
            start_opacity=1.0,
        )

    def _on_tick(self) -> None:
        if self._fade_running:
            elapsed = time.monotonic() - self._fade_started
            progress = min(1.0, elapsed / max(1e-3, self._fade_duration / 1000.0))
            eased = progress * progress * (3.0 - 2.0 * progress)
            self._fade_opacity = (
                self._fade_from + (self._fade_target - self._fade_from) * eased
            )
            if progress >= 1.0:
                self._fade_opacity = self._fade_target
                self._fade_running = False
                callback = self._fade_callback
                self._fade_callback = None
                if callback is not None:
                    callback()
        self.update()
        self._maybe_hide()

    # ---- 居中提示 -----------------------------------------------------
    def show_message(
        self,
        text: str,
        *,
        success: bool = False,
        duration_ms: int = PROMPT_DEFAULT_MS,
        subtitle: Optional[str] = None,
    ) -> None:
        self._message = str(text)
        self._subtitle = (
            subtitle
            if subtitle is not None
            else (
                "正在进入外星花园互动…"
                if success
                else "请调整画纸后再按 Space / Enter"
            )
        )
        self._accent = QColor(96, 232, 184) if success else QColor(255, 92, 92)
        self._message_started = time.monotonic()
        self._message_visible = True
        self._message_timer.start(max(600, int(duration_ms)))
        self._ensure_visible()
        self._anim_timer.start()
        self.update()

    def clear_message(self) -> None:
        self._message_timer.stop()
        self._message_visible = False
        self.update()
        self._maybe_hide()

    @property
    def message(self) -> str:
        return self._message

    @property
    def message_visible(self) -> bool:
        return self._message_visible

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)

        if self._fade_opacity > 0.001:
            fade = QColor(self._fade_color)
            fade.setAlphaF(self._fade_opacity)
            painter.fillRect(self.rect(), fade)

        if self._message_visible:
            self._paint_message(painter)
        painter.end()

    def _paint_message(self, painter: QPainter) -> None:
        bounds = QRectF(self.rect())
        elapsed = max(0.0, time.monotonic() - self._message_started)
        pulse = (math.sin(elapsed * 4.2) + 1.0) / 2.0
        bob = math.sin(elapsed * 3.4) * 5.0

        width = min(bounds.width() * 0.78, 980.0)
        height = min(max(150.0, bounds.height() * 0.23), 210.0)
        panel = QRectF(
            bounds.center().x() - width / 2.0,
            bounds.center().y() - height / 2.0 + bob,
            width,
            height,
        )

        glow = QRadialGradient(panel.center(), width * 0.58)
        glow.setColorAt(
            0.0,
            QColor(
                self._accent.red(),
                self._accent.green(),
                self._accent.blue(),
                int(80 + 55 * pulse),
            ),
        )
        glow.setColorAt(
            1.0,
            QColor(
                self._accent.red(),
                self._accent.green(),
                self._accent.blue(),
                0,
            ),
        )
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(glow))
        painter.drawEllipse(panel.center(), width * 0.52, height * 0.92)

        painter.setBrush(QBrush(QColor(4, 9, 13, 236)))
        border = QColor(self._accent)
        border.setAlpha(int(150 + 95 * pulse))
        painter.setPen(QPen(border, 2.0 + 1.4 * pulse))
        painter.drawRoundedRect(panel, 24, 24)

        icon_center = QPointF(panel.left() + 72.0, panel.center().y())
        icon_fill = QColor(self._accent)
        icon_fill.setAlpha(44)
        painter.setBrush(QBrush(icon_fill))
        painter.setPen(QPen(self._accent, 2.0))
        painter.drawEllipse(icon_center, 34.0, 34.0)
        painter.setFont(self._icon_font)
        painter.setPen(self._accent)
        painter.drawText(
            QRectF(icon_center.x() - 34, icon_center.y() - 34, 68, 68),
            Qt.AlignCenter,
            "✓" if self._accent.green() > 180 else "!",
        )

        text_left = panel.left() + 126.0
        text_width = panel.width() - 156.0
        painter.setFont(self._title_font)
        painter.setPen(QColor(242, 252, 248))
        painter.drawText(
            QRectF(text_left, panel.top() + 24.0, text_width, height - 80.0),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            self._message,
        )

        painter.setFont(self._subtitle_font)
        painter.setPen(
            QColor(
                self._accent.red(),
                self._accent.green(),
                self._accent.blue(),
                220,
            )
        )
        painter.drawText(
            QRectF(text_left, panel.bottom() - 46.0, text_width, 26.0),
            Qt.AlignLeft | Qt.AlignVCenter,
            self._subtitle,
        )


class StandbyWindow(QWidget):
    """全屏无边框展示窗口：待机视频 / 摄像头扫描 / 场景联动三态切换。"""

    def __init__(
        self,
        video_path: Optional[Path] = None,
        volume: int = 0,
        windowed: bool = False,
        force_fallback: bool = False,
        camera_index: int = 0,
        camera_width: int = 1280,
        camera_height: int = 720,
        start_camera: bool = False,
        camera_enabled: bool = True,
        start_preview: bool = False,
        capture_dir: Optional[Path] = None,
        tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
        min_paper_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
        scene_seconds: float = DEFAULT_SCENE_SECONDS,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("外星花园 · 交互展示")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setStyleSheet("background-color: #000000;")
        self.setMinimumSize(800, 450)
        self.setFocusPolicy(Qt.StrongFocus)

        self._windowed = windowed
        self._video_path = video_path
        self._volume = max(0, min(100, volume))
        self._force_fallback = force_fallback
        self._player: Optional[QMediaPlayer] = None
        self._playlist: Optional[QMediaPlaylist] = None
        self._video_widget: Optional[QVideoWidget] = None

        self._camera_index = camera_index
        self._camera_width = camera_width
        self._camera_height = camera_height
        self._start_camera_on_launch = start_camera
        self._start_preview = bool(start_preview)
        self._camera_enabled = bool(camera_enabled and SCANNER_AVAILABLE)
        self._tilt_threshold = float(tilt_threshold)
        self._min_paper_ratio = float(min_paper_ratio)
        self._scene_seconds = max(2.0, float(scene_seconds))
        self._capture_dir = Path(capture_dir) if capture_dir is not None else CAPTURE_DIR

        self._vision_worker: Optional[VisionWorker] = None
        self._camera_error = ""
        self._last_frame: Any = None
        self._last_analysis: Any = None
        self._last_capture_path: Optional[Path] = None
        self._candidate_code: Optional[str] = None
        self._ready_streak = 0
        self._last_trigger_at = 0.0
        self._last_button_at = 0.0
        self._trigger_count = 0
        self._preview_mode = False
        self._state = ShowcaseState.STANDBY
        self._shutter_wav: Optional[bytes] = None

        self._scene_timer = QTimer(self)
        self._scene_timer.setSingleShot(True)
        self._scene_timer.timeout.connect(self._finish_scene)

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setStackingMode(QStackedLayout.StackOne)

        self._idle_canvas = IdleCanvas(self)
        self._stack.addWidget(self._idle_canvas)

        self._camera_canvas = CameraCanvas(self)
        self._stack.addWidget(self._camera_canvas)
        self._scene_canvas = SceneCanvas(self)
        self._stack.addWidget(self._scene_canvas)

        if self._force_fallback or self._video_path is None:
            self._stack.setCurrentWidget(self._idle_canvas)
        elif not MULTIMEDIA_AVAILABLE:
            LOGGER.warning(
                "QtMultimedia 不可用（%s），回退到呼吸文字待机页",
                MULTIMEDIA_IMPORT_ERROR,
            )
            self._stack.setCurrentWidget(self._idle_canvas)
        else:
            self._setup_video(self._video_path)

        self._standby_widget = self._stack.currentWidget()
        self._overlay = ShowcaseOverlay(self)
        self._overlay.sync_geometry(self)

    # ---- 视频播放 -----------------------------------------------------
    def _setup_video(self, video_path: Path) -> None:
        self._video_widget = QVideoWidget(self)
        self._video_widget.setStyleSheet("background-color: #000000;")
        self._video_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._video_widget.setAspectRatioMode(Qt.KeepAspectRatio)
        self._stack.addWidget(self._video_widget)

        self._player = QMediaPlayer(self, QMediaPlayer.VideoSurface)
        self._playlist = QMediaPlaylist(self)
        self._playlist.setPlaybackMode(QMediaPlaylist.Loop)
        self._playlist.addMedia(QMediaContent(QUrl.fromLocalFile(str(video_path))))
        self._player.setPlaylist(self._playlist)
        self._player.setVideoOutput(self._video_widget)
        self._player.setVolume(self._volume)

        # Qt5.15 同时存在旧 error 信号和新 errorOccurred 信号，按可用性连接
        if hasattr(self._player, "errorOccurred"):
            self._player.errorOccurred.connect(self._on_player_error)
        else:
            self._player.error.connect(self._on_player_error)
        self._player.mediaStatusChanged.connect(self._on_media_status)
        self._stack.setCurrentWidget(self._video_widget)

    def _on_player_error(self, *_args) -> None:
        message = self._player.errorString() if self._player is not None else "未知错误"
        self._activate_fallback(f"待机视频播放失败：{message}")

    def _on_media_status(self, status) -> None:
        if status == QMediaPlayer.InvalidMedia:
            self._activate_fallback("待机视频格式无效或文件损坏")
        elif status == QMediaPlayer.EndOfMedia:
            LOGGER.debug("待机视频到达结尾，播放列表将自动循环播放")

    def _activate_fallback(self, reason: str) -> None:
        if self._stack.currentWidget() is self._idle_canvas:
            return
        LOGGER.warning("切换至呼吸文字待机页：%s", reason)
        if self._player is not None:
            self._player.stop()
        self._standby_widget = self._idle_canvas
        # 若当前正在摄像头预览或互动页，只更新待机兜底，不打断当前画面
        if self._stack.currentWidget() is self._video_widget:
            self._stack.setCurrentWidget(self._idle_canvas)

    # ---- 摄像头扫描 ---------------------------------------------------
    def _pause_standby_video(self) -> None:
        if self._player is not None and self._stack.currentWidget() is self._video_widget:
            self._player.pause()

    def _resume_standby_video(self) -> None:
        if (
            self._player is not None
            and self._video_widget is not None
            and self._standby_widget is self._video_widget
            and self._stack.currentWidget() is self._video_widget
        ):
            self._player.play()

    def _show_camera(self) -> None:
        self._pause_standby_video()
        self._scene_timer.stop()
        self._scene_canvas.clear_scene()
        self._preview_mode = True
        self._stack.setCurrentWidget(self._camera_canvas)
        self._camera_canvas.setFocus()

    def _show_standby(self) -> None:
        self._scene_timer.stop()
        self._scene_canvas.clear_scene()
        self._preview_mode = False
        self._stack.setCurrentWidget(self._standby_widget)
        self._resume_standby_video()
        self.setFocus()

    def _start_camera(self, *, show_preview: bool = False) -> None:
        if not self._camera_enabled:
            message = "视觉模块不可用" if not SCANNER_AVAILABLE else "摄像头功能已禁用"
            LOGGER.warning(message)
            if not SCANNER_AVAILABLE:
                LOGGER.warning("scanner 导入失败：%s", SCANNER_IMPORT_ERROR)
            if show_preview:
                self._camera_canvas.set_error(message)
                self._preview_mode = True
                self._stack.setCurrentWidget(self._camera_canvas)
            return

        if self._vision_worker is not None:
            if show_preview:
                self._show_camera()
            return

        config = CameraConfig(
            index=self._camera_index,
            width=self._camera_width,
            height=self._camera_height,
        )
        worker = VisionWorker(
            config,
            tilt_threshold=self._tilt_threshold,
            min_paper_ratio=self._min_paper_ratio,
        )
        worker.frame_ready.connect(self._on_frame_ready)
        worker.camera_opened.connect(self._on_camera_opened)
        worker.camera_error.connect(self._on_camera_error)
        worker.finished.connect(self._on_vision_finished)

        self._vision_worker = worker
        self._camera_error = ""
        self._camera_canvas.clear()
        worker.start()
        if show_preview:
            self._show_camera()
        LOGGER.info(
            "启动后台摄像头扫描：index=%s，请求分辨率=%sx%s，倾斜阈值=±%.1f°",
            self._camera_index,
            self._camera_width,
            self._camera_height,
            self._tilt_threshold,
        )

    def _stop_camera(self) -> None:
        worker = self._vision_worker
        self._vision_worker = None
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        self._camera_canvas.clear()
        self._ready_streak = 0
        self._candidate_code = None
        self._last_frame = None
        self._last_analysis = None
        self._show_standby()
        LOGGER.info("摄像头扫描已停止")

    def _toggle_camera(self) -> None:
        if self._state is ShowcaseState.INTERACTIVE:
            self._finish_scene()
            return
        if self._state is ShowcaseState.RESET:
            return
        if not self._camera_enabled:
            self._overlay.show_message("摄像头未启用，无法进入扫描预览", success=False)
            return
        if self._vision_worker is None:
            self._start_camera(show_preview=True)
        elif self._preview_mode:
            self._show_standby()
        else:
            self._show_camera()

    def _on_camera_opened(self, message: str) -> None:
        self._camera_error = ""
        LOGGER.info("视觉模块就绪：%s", message)

    def _on_camera_error(self, message: str) -> None:
        self._camera_error = message
        self._camera_canvas.set_error(message)
        LOGGER.error("摄像头异常：%s", message)
        if self._preview_mode:
            self._overlay.show_message(
                "摄像头连接异常，请检查高拍仪",
                success=False,
                subtitle=message,
            )

    def _on_vision_finished(self) -> None:
        worker = self._vision_worker
        if worker is not None and not worker.isRunning():
            worker.deleteLater()
            self._vision_worker = None
            if self._camera_error:
                self._camera_canvas.set_error(self._camera_error)
            LOGGER.info("视觉线程已结束")


    # ---- 按键 / 外部大按钮触发 ----------------------------------------
    @property
    def state(self) -> str:
        """当前大屏状态：``standby`` / ``interactive`` / ``reset``。"""
        return self._state.value

    @property
    def last_capture_path(self) -> Optional[Path]:
        return self._last_capture_path

    def _on_frame_ready(self, frame: Any, analysis: Any) -> None:
        # 即使当前显示待机视频，也持续保留最新一帧与分析结果供按键拍照使用。
        self._last_frame = frame
        self._last_analysis = analysis
        self._camera_canvas.set_frame(frame, analysis)

    def _reject_trigger(self, message: str) -> bool:
        self._overlay.show_message(message, success=False)
        QApplication.beep()
        LOGGER.info("按键触发被拒绝：%s", message)
        return False

    def trigger_capture(self) -> bool:
        """供 Space / Enter（或外部 USB 大按钮）调用的拍照入口。

        Returns:
            True 表示校验通过并已进入互动状态；False 表示被拒绝。
        """
        if self._state is ShowcaseState.INTERACTIVE:
            self._overlay.show_message(
                "作品正在大屏展示，请稍候",
                success=True,
                duration_ms=1400,
            )
            return False
        if self._state is ShowcaseState.RESET:
            return False

        if not self._camera_enabled:
            return self._reject_trigger("摄像头未启用，无法拍照")

        frame = self._last_frame
        analysis = self._last_analysis
        if frame is None or analysis is None:
            return self._reject_trigger("扫描台尚未就绪，请将画作放入扫描区后再按")

        is_straight = bool(getattr(analysis, "is_straight", False))
        code_id = str(
            getattr(analysis, "code_id", "")
            or getattr(analysis, "code_data", "")
            or ""
        ).strip()

        if not is_straight:
            pose = getattr(analysis, "pose", None)
            if pose is not None and getattr(pose, "found", False):
                angle = abs(float(getattr(pose, "angle_deg", 0.0)))
                message = f"图纸未摆正，请调整后再按（偏离 {angle:.1f}°）"
            else:
                message = "未识别到完整图纸，请将画作平放入扫描区"
            return self._reject_trigger(message)

        if not code_id:
            return self._reject_trigger("未识别到二维码，请将二维码对准扫描区后再按")

        # 防止外接按钮抖动 / 键盘重复事件在极短时间内重复触发
        now = time.monotonic()
        if now - self._last_button_at < BUTTON_DEBOUNCE_SECONDS:
            return False
        self._last_button_at = now

        artwork = self._extract_artwork(frame, analysis)
        if artwork is None or getattr(artwork, "size", 0) == 0:
            return self._reject_trigger("未能截取画作，请调整图纸位置后重试")

        snapshot = bgr_to_qimage(artwork)
        capture_path = self._save_artwork(snapshot, code_id)
        if capture_path is None:
            return self._reject_trigger("画作保存失败，请检查 captures 目录权限")

        self._last_capture_path = capture_path
        self._play_shutter_feedback()
        self._overlay.show_message(
            "拍照成功，正在进入外星花园…",
            success=True,
            duration_ms=900,
        )
        self._trigger_scene(snapshot, analysis, code_id, capture_path)
        return True

    def _extract_artwork(self, frame: Any, analysis: Any) -> Any:
        """优先透视校正旋转纸张，失败时退回外接矩形裁切。"""
        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        pose = getattr(analysis, "pose", None)
        image: Any = None
        quad = (
            getattr(pose, "quad", None)
            if pose is not None and getattr(pose, "found", False)
            else None
        )
        if cv2 is not None and quad is not None and len(quad) == 4:
            points = np.asarray(quad, dtype=np.float32)
            if points.shape == (4, 2):
                top = float(np.linalg.norm(points[0] - points[1]))
                bottom = float(np.linalg.norm(points[3] - points[2]))
                left = float(np.linalg.norm(points[0] - points[3]))
                right = float(np.linalg.norm(points[1] - points[2]))
                width = max(top, bottom)
                height = max(left, right)
                if width >= 80.0 and height >= 80.0:
                    target = np.array(
                        [
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1],
                            [0, height - 1],
                        ],
                        dtype=np.float32,
                    )
                    try:
                        matrix = cv2.getPerspectiveTransform(points, target)
                        warped = cv2.warpPerspective(
                            frame,
                            matrix,
                            (int(round(width)), int(round(height))),
                            flags=cv2.INTER_LINEAR,
                        )
                        if getattr(warped, "size", 0) > 0:
                            image = warped
                    except Exception as exc:
                        LOGGER.debug("透视裁切失败，退回矩形裁切：%s", exc)

        if image is None and pose is not None and getattr(pose, "rect", None):
            x, y, width, height = (int(value) for value in pose.rect)
            if width > 24 and height > 24:
                frame_height, frame_width = frame.shape[:2]
                x0 = max(0, x)
                y0 = max(0, y)
                x1 = min(frame_width, x + width)
                y1 = min(frame_height, y + height)
                if x1 > x0 and y1 > y0:
                    image = frame[y0:y1, x0:x1].copy()

        if image is None or getattr(image, "size", 0) == 0:
            image = frame.copy()
        return image

    def _save_artwork(self, image: QImage, code_id: str) -> Optional[Path]:
        """把观众画作保存为 JPG，返回实际文件路径。"""
        if image.isNull():
            return None
        try:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            LOGGER.error("创建 captures 目录失败：%s", exc)
            return None

        safe_code = re.sub(r"[^0-9A-Za-z_-]+", "_", code_id or "").strip("_")
        safe_code = safe_code[:48] or "artwork"
        stamp = time.strftime("%Y%m%d_%H%M%S")
        target = self._capture_dir / f"artwork_{stamp}_{safe_code}.jpg"
        for attempt in range(2):
            path = target
            if attempt > 0:
                path = target.with_name(
                    f"{target.stem}_{int(time.time() * 1000) % 1000}{target.suffix}"
                )
            try:
                if image.save(str(path), "JPG", 92):
                    LOGGER.info("画作已保存：%s", path)
                    return path
            except Exception as exc:
                LOGGER.warning("保存画作失败：%s", exc)
        LOGGER.error("无法写入画作文件：%s", target)
        return None

    def _play_shutter_feedback(self) -> None:
        """播放快门声 + 全屏白闪。"""
        self._overlay.flash(color=(255, 255, 255), duration_ms=SHUTTER_FLASH_MS)
        if self._shutter_wav is None:
            try:
                self._shutter_wav = build_shutter_wav()
            except Exception as exc:
                LOGGER.debug("合成快门声失败：%s", exc)
        if self._shutter_wav and sys.platform.startswith("win"):
            try:
                import winsound

                winsound.PlaySound(
                    self._shutter_wav,
                    winsound.SND_MEMORY | winsound.SND_ASYNC,
                )
                return
            except Exception as exc:
                LOGGER.debug("播放快门声失败：%s", exc)
        QApplication.beep()

    def _trigger_scene(
        self,
        snapshot: QImage,
        analysis: Any,
        code: str,
        capture_path: Optional[Path] = None,
    ) -> None:
        if self._state is not ShowcaseState.STANDBY:
            return
        theme = resolve_scene_theme(code)
        self._state = ShowcaseState.INTERACTIVE
        self._preview_mode = False
        self._scene_canvas.set_scene(
            theme,
            code,
            snapshot,
            duration=self._scene_seconds,
        )
        self._pause_standby_video()
        self._stack.setCurrentWidget(self._scene_canvas)
        self.setFocus()
        self._scene_timer.start(int(self._scene_seconds * 1000))
        self._trigger_count += 1
        self._last_trigger_at = time.monotonic()
        LOGGER.info(
            "按键拍照 #%d：二维码=%s → 主题=%s，画作=%s（%.0f 秒后淡出）",
            self._trigger_count,
            code,
            theme.title,
            capture_path or "未保存",
            self._scene_seconds,
        )

    def _finish_scene(self) -> None:
        """互动展示结束：先淡出到黑场，再切回待机视频。"""
        self._scene_timer.stop()
        if self._state is not ShowcaseState.INTERACTIVE:
            return
        self._state = ShowcaseState.RESET
        self._overlay.clear_message()
        LOGGER.info("互动展示结束，开始淡出并返回待机")
        self._overlay.fade_to(
            1.0,
            SCENE_FADE_OUT_MS,
            color=(0, 0, 0),
            on_finished=self._complete_reset,
        )

    def _complete_reset(self) -> None:
        if self._state is not ShowcaseState.RESET:
            return
        self._show_standby()
        self._state = ShowcaseState.STANDBY
        self._overlay.fade_to(0.0, SCENE_FADE_IN_MS, color=(0, 0, 0))
        LOGGER.info("已返回待机视频循环，等待下一位观众")

    def selftest_summary(self) -> dict:
        """给 --selftest 使用的运行摘要，便于终端断言摄像头与联动状态。"""
        analysis = self._last_analysis
        return {
            "scanner_available": SCANNER_AVAILABLE,
            "camera_requested": self._start_camera_on_launch,
            "camera_enabled": self._camera_enabled,
            "camera_preview": self._preview_mode,
            "frames": self._camera_canvas.frames,
            "triggers": self._trigger_count,
            "state": self._state.value,
            "last_capture": str(self._last_capture_path) if self._last_capture_path else None,
            "last_status": getattr(getattr(analysis, "status", None), "value", None),
            "last_qr": getattr(analysis, "code_data", None),
            "camera_error": self._camera_error,
            "current_view": type(self._stack.currentWidget()).__name__,
        }


    # ---- 生命周期与交互 -----------------------------------------------
    def start(self) -> None:
        """显示窗口并开始待机展示。"""
        if self._windowed:
            self.setCursor(Qt.ArrowCursor)
            self.resize(1280, 720)
            self.show()
        else:
            self.setCursor(Qt.BlankCursor)
            self.showFullScreen()
        self.raise_()
        self.activateWindow()
        self.setFocus()

        playing_video = (
            self._player is not None
            and self._video_widget is not None
            and self._stack.currentWidget() is self._video_widget
        )
        if playing_video:
            self._player.play()
            LOGGER.info("开始循环播放待机视频：%s", self._video_path)
        else:
            LOGGER.info("待机页已启用：黑色背景 + 呼吸文字")

        if self._start_camera_on_launch:
            self._start_camera(show_preview=self._start_preview)
        else:
            LOGGER.info("摄像头未启用，仅待机视频；Space / Enter 不会触发拍照")

        self._sync_overlay_geometry()

    def _sync_overlay_geometry(self) -> None:
        overlay = getattr(self, "_overlay", None)
        if overlay is not None:
            overlay.sync_geometry(self)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._sync_overlay_geometry()

    def moveEvent(self, event) -> None:  # noqa: N802
        super().moveEvent(event)
        self._sync_overlay_geometry()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._sync_overlay_geometry()

    def hideEvent(self, event) -> None:  # noqa: N802
        overlay = getattr(self, "_overlay", None)
        if overlay is not None:
            overlay.hide()
        super().hideEvent(event)

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.setCursor(Qt.ArrowCursor)
            self.showNormal()
            self.resize(1280, 720)
        else:
            self.setCursor(Qt.BlankCursor)
            self.showFullScreen()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            if not event.isAutoRepeat():
                self.trigger_capture()
            event.accept()
            return
        if event.key() == Qt.Key_C:
            self._toggle_camera()
            event.accept()
            return
        if event.key() == Qt.Key_Escape:
            LOGGER.info("收到 ESC，退出程序")
            self.close()
            event.accept()
            return
        if event.key() == Qt.Key_F11:
            self._toggle_fullscreen()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        LOGGER.info("正在关闭展示窗口")
        self._scene_timer.stop()
        overlay = getattr(self, "_overlay", None)
        if overlay is not None:
            overlay.hide()
            overlay.close()
        worker = self._vision_worker
        self._vision_worker = None
        if worker is not None:
            worker.stop()
            worker.deleteLater()
        if self._player is not None:
            self._player.stop()
            self._player.deleteLater()
            self._player = None
        if self._playlist is not None:
            self._playlist.clear()
            self._playlist.deleteLater()
            self._playlist = None
        self._idle_canvas.hide()
        super().closeEvent(event)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外星花园 · 交互展示（阶段三：按键触发与视频流转）")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="待机视频路径；缺省时自动扫描 assets/videos",
    )
    parser.add_argument("--windowed", action="store_true", help="以 1280x720 窗口运行，便于调试")
    parser.add_argument("--no-video", action="store_true", help="强制关闭待机视频，使用呼吸文字待机页")
    parser.add_argument(
        "--volume",
        type=int,
        default=0,
        help="待机视频音量 0-100（默认 0，静音）",
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号（默认 0，笔记本内置摄像头）")
    parser.add_argument("--width", type=int, default=1280, help="摄像头采集宽度（默认 1280）")
    parser.add_argument("--height", type=int, default=720, help="摄像头采集高度（默认 720）")
    parser.add_argument(
        "--start-camera",
        action="store_true",
        help="兼容旧版本参数；摄像头现已默认开启",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="禁用摄像头（只保留待机展示）",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help="启动时直接显示摄像头预览（默认停留在待机视频，按 C 可切换预览）",
    )
    parser.add_argument(
        "--tilt-threshold",
        type=float,
        default=DEFAULT_TILT_THRESHOLD_DEG,
        help=f"图纸允许倾斜角，单位度（默认 {DEFAULT_TILT_THRESHOLD_DEG:.0f}）",
    )
    parser.add_argument(
        "--min-paper-ratio",
        type=float,
        default=DEFAULT_MIN_PAPER_AREA_RATIO,
        help=f"图纸最小面积占比（默认 {DEFAULT_MIN_PAPER_AREA_RATIO:.2f}）",
    )
    parser.add_argument(
        "--scene-seconds",
        type=float,
        default=DEFAULT_SCENE_SECONDS,
        help=f"联动画面停留秒数（默认 {DEFAULT_SCENE_SECONDS:.0f}）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认 INFO）",
    )
    parser.add_argument("--selftest", action="store_true", help="启动后自动退出并输出运行摘要")
    parser.add_argument(
        "--selftest-seconds",
        type=float,
        default=3.0,
        help="--selftest 运行时长（默认 3 秒）",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # 高分屏下保持清晰，必须在 QApplication 创建之前设置
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Alien Garden Interactive Showcase")
    app.setOrganizationName("Alien Garden")

    video_path = None if args.no_video else find_standby_video(args.video)
    if video_path is not None:
        LOGGER.info("待机视频：%s", video_path)
    else:
        LOGGER.info(
            "未找到可用的待机视频，使用内置呼吸文字待机页（视频目录：%s）",
            VIDEO_DIR,
        )

    if not SCANNER_AVAILABLE:
        LOGGER.warning("视觉模块不可用：%s", SCANNER_IMPORT_ERROR)
    elif args.no_camera:
        LOGGER.info("摄像头已通过 --no-camera 禁用")
    else:
        LOGGER.info("启动后立即进入摄像头扫描（index=%s）", args.camera)

    window = StandbyWindow(
        video_path=video_path,
        volume=args.volume,
        windowed=args.windowed,
        force_fallback=args.no_video,
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        start_camera=not args.no_camera,
        camera_enabled=not args.no_camera,
        start_preview=args.camera_preview,
        tilt_threshold=args.tilt_threshold,
        min_paper_ratio=args.min_paper_ratio,
        scene_seconds=args.scene_seconds,
    )
    window.start()

    if args.selftest:
        delay_ms = max(300, int(args.selftest_seconds * 1000))
        QTimer.singleShot(delay_ms, window.close)
        LOGGER.info("自检模式：%.1f 秒后自动退出", delay_ms / 1000.0)

    exit_code = app.exec_()

    if args.selftest:
        summary = window.selftest_summary()
        print("SELFTEST_SUMMARY " + json.dumps(summary, ensure_ascii=False))
        ok = True
        if not args.no_camera:
            ok = summary["frames"] > 0 and not summary["camera_error"]
        print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
        return 0 if ok else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
