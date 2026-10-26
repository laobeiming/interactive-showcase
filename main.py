"""外星花园交互展示程序 —— 阶段二：视觉检测与画面联动。

用法：
    python main.py                              # 待机展示（自动查找待机视频）
    python main.py --windowed                   # 1280x720 窗口调试
    python main.py --windowed --no-video --start-camera
                                                # 直接用笔记本摄像头测试
    python main.py --windowed --no-video --start-camera --camera 0
    python main.py --no-video --start-camera --tilt-threshold 8

快捷键：
    C    打开 / 关闭摄像头实时扫描（联动后按 C 返回扫描页）
    ESC  退出程序
    F11  全屏 / 窗口切换（调试用）

联动逻辑：
    摄像头画面中同时满足「检测到图纸 + 倾斜角 ≤ 阈值 + 二维码识别成功」并稳定
    若干帧后，自动切入场景联动页；图纸倾斜超阈值时只在扫描页提示校正方向，
    不会误触发大屏。默认 8 秒后自动返回扫描页，可重复触发。

待机视频仅作为待机背景保留；没有视频时自动使用黑色呼吸文字待机页。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
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
    DEFAULT_TILT_THRESHOLD_DEG = 8.0
    SCANNER_AVAILABLE = False
    SCANNER_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "assets" / "videos"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".webm", ".m4v"}
PREFERRED_VIDEO_KEYWORDS = ("standby", "idle", "waiting", "daiji", "待机")

# 联动触发参数
TRIGGER_STABLE_FRAMES = 3
TRIGGER_COOLDOWN_SECONDS = 3.0
DEFAULT_SCENE_SECONDS = 8.0
CAMERA_ANALYSIS_INTERVAL = 0.12
CAMERA_FRAME_INTERVAL = 1.0 / 24.0

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
    """确定待机视频：优先使用显式路径，否则扫描 assets/videos 目录。"""
    if explicit_path is not None:
        path = explicit_path.expanduser()
        if not path.is_absolute():
            path = (BASE_DIR / path).resolve()
        if path.is_file() and path.stat().st_size > 0:
            return path
        LOGGER.warning("指定的视频不可用：%s", path)
        return None

    if not VIDEO_DIR.is_dir():
        return None

    candidates = [
        item
        for item in sorted(VIDEO_DIR.iterdir())
        if item.is_file()
        and item.suffix.lower() in VIDEO_SUFFIXES
        and item.stat().st_size > 0
    ]
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
            return QColor(96, 232, 184), "识别成功", "图纸水平 · 二维码匹配 · 正在联动大屏"
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

        accent = QColor(96, 232, 184) if not pose.needs_correction else QColor(255, 184, 92)
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
            angle_color = QColor(96, 232, 184) if not pose.needs_correction else QColor(255, 184, 92)
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
            "C 关闭摄像头   ·   ESC 退出",
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
        left = bounds.width() * 0.075
        block_width = bounds.width() * 0.50

        painter.setFont(self._kicker_font)
        painter.setPen(accent)
        painter.drawText(
            QRectF(left, bounds.height() * 0.20, block_width, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "SCENE LINKED  /  二维码已匹配",
        )

        title_size = int(max(30, min(78, bounds.height() * 0.092)))
        title_font = QFont(self._family, title_size, QFont.Bold)
        painter.setFont(title_font)
        painter.setPen(QColor(240, 252, 248))
        painter.drawText(
            QRectF(left, bounds.height() * 0.27, block_width, title_size * 1.35),
            Qt.AlignLeft | Qt.AlignVCenter,
            theme.title,
        )

        painter.setFont(self._body_font)
        painter.setPen(QColor(176, 206, 198))
        painter.drawText(
            QRectF(left, bounds.height() * 0.27 + title_size * 1.4, block_width, 62),
            Qt.AlignLeft | Qt.AlignTop | Qt.TextWordWrap,
            theme.subtitle,
        )

        pill_rect = QRectF(left, bounds.height() * 0.72, min(block_width, 440), 38)
        painter.setBrush(QBrush(QColor(8, 22, 20, 200)))
        painter.setPen(QPen(QColor(accent.red(), accent.green(), accent.blue(), 120), 1.2))
        painter.drawRoundedRect(pill_rect, 19, 19)

        painter.setFont(self._mono_font)
        painter.setPen(QColor(214, 242, 232))
        qr_label = self._qr_data or theme.key.upper()
        qr_label = QFontMetrics(self._mono_font).elidedText(qr_label, Qt.ElideMiddle, int(pill_rect.width() - 40))
        painter.drawText(pill_rect.adjusted(18, 0, -14, 0), Qt.AlignLeft | Qt.AlignVCenter, f"ID  {qr_label}")

        painter.setFont(self._small_font)
        painter.setPen(QColor(142, 174, 168))
        painter.drawText(
            QRectF(left, bounds.height() * 0.72 + 50, block_width, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            f"按 C 返回扫描台  ·  {self.remaining_seconds():.0f} 秒后自动返回",
        )


    def _draw_snapshot_card(
        self,
        painter: QPainter,
        bounds: QRectF,
        theme: SceneTheme,
        accent: QColor,
    ) -> None:
        card_width = min(bounds.width() * 0.30, bounds.height() * 0.54)
        card_height = card_width * 1.22
        card = QRectF(
            bounds.width() * 0.635,
            (bounds.height() - card_height) / 2.0 - 16,
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
        painter.drawText(
            QRectF(card.left() + 20, card.bottom() - 58, card.width() - 40, 40),
            Qt.AlignLeft | Qt.AlignVCenter | Qt.TextWordWrap,
            f"主题 · {theme.title}\n已投射至大屏",
        )

    def _draw_progress(self, painter: QPainter, bounds: QRectF, accent: QColor) -> None:
        painter.setFont(self._mono_font)
        painter.setPen(QColor(120, 156, 150))
        painter.drawText(
            QRectF(bounds.width() * 0.075, 24, bounds.width() * 0.40, 22),
            Qt.AlignLeft | Qt.AlignVCenter,
            "ALIEN GARDEN  /  IMMERSIVE SHOWCASE",
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
        self._camera_enabled = bool(camera_enabled and SCANNER_AVAILABLE)
        self._tilt_threshold = float(tilt_threshold)
        self._min_paper_ratio = float(min_paper_ratio)
        self._scene_seconds = max(2.0, float(scene_seconds))

        self._vision_worker: Optional[VisionWorker] = None
        self._camera_error = ""
        self._last_analysis: Any = None
        self._candidate_code: Optional[str] = None
        self._ready_streak = 0
        self._last_trigger_at = 0.0
        self._trigger_count = 0

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
        self._stack.setCurrentWidget(self._camera_canvas)
        self._camera_canvas.setFocus()

    def _show_standby(self) -> None:
        self._scene_timer.stop()
        self._scene_canvas.clear_scene()
        self._stack.setCurrentWidget(self._standby_widget)
        self._resume_standby_video()

    def _start_camera(self) -> None:
        if not self._camera_enabled:
            message = "视觉模块不可用" if not SCANNER_AVAILABLE else "摄像头功能已禁用"
            self._camera_canvas.set_error(message)
            self._stack.setCurrentWidget(self._camera_canvas)
            LOGGER.warning(message)
            if not SCANNER_AVAILABLE:
                LOGGER.warning("scanner 导入失败：%s", SCANNER_IMPORT_ERROR)
            return

        if self._vision_worker is not None:
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
        self._show_camera()
        worker.start()
        LOGGER.info(
            "启动摄像头扫描：index=%s，请求分辨率=%sx%s，倾斜阈值=±%.1f°",
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
        self._last_analysis = None
        self._show_standby()
        LOGGER.info("摄像头扫描已停止")

    def _toggle_camera(self) -> None:
        if self._stack.currentWidget() is self._scene_canvas:
            self._finish_scene()
            return
        if self._vision_worker is None:
            self._start_camera()
        else:
            self._stop_camera()

    def _on_camera_opened(self, message: str) -> None:
        self._camera_error = ""
        LOGGER.info("视觉模块就绪：%s", message)

    def _on_camera_error(self, message: str) -> None:
        self._camera_error = message
        self._camera_canvas.set_error(message)
        LOGGER.error("摄像头异常：%s", message)

    def _on_vision_finished(self) -> None:
        worker = self._vision_worker
        if worker is not None and not worker.isRunning():
            worker.deleteLater()
            self._vision_worker = None
            if self._camera_error:
                self._camera_canvas.set_error(self._camera_error)
            LOGGER.info("视觉线程已结束")


    # ---- 联动逻辑 -----------------------------------------------------
    def _on_frame_ready(self, frame: Any, analysis: Any) -> None:
        self._camera_canvas.set_frame(frame, analysis)
        self._last_analysis = analysis
        if analysis is None:
            return
        if self._stack.currentWidget() is not self._camera_canvas:
            return
        self._update_linkage(frame, analysis)

    def _reset_trigger_streak(self) -> None:
        self._ready_streak = 0
        self._candidate_code = None

    def _update_linkage(self, frame: Any, analysis: Any) -> None:
        """只有「二维码 + 图纸水平」稳定成立时才允许联动，倾斜时只提示校正。"""
        status_value = getattr(getattr(analysis, "status", None), "value", "")
        code = getattr(analysis, "code_data", None)

        if status_value == "ready" and code:
            if code == self._candidate_code:
                self._ready_streak += 1
            else:
                self._candidate_code = code
                self._ready_streak = 1

            if self._ready_streak >= TRIGGER_STABLE_FRAMES:
                now = time.monotonic()
                if now - self._last_trigger_at >= TRIGGER_COOLDOWN_SECONDS:
                    self._last_trigger_at = now
                    self._trigger_scene(frame, analysis, code)
            return

        if status_value == "tilted":
            self._reset_trigger_streak()
            pose = getattr(analysis, "pose", None)
            if pose is not None:
                now = time.monotonic()
                if now - getattr(self, "_last_tilt_log", 0.0) > 2.0:
                    self._last_tilt_log = now
                    LOGGER.info(
                        "图纸倾斜 %+.1f°（阈值 ±%.1f°）：%s",
                        pose.angle_deg,
                        pose.tilt_threshold,
                        pose.correction_hint,
                    )
            return

        self._reset_trigger_streak()

    def _trigger_scene(self, frame: Any, analysis: Any, code: str) -> None:
        theme = resolve_scene_theme(code)
        snapshot = bgr_to_qimage(frame)
        pose = getattr(analysis, "pose", None)
        if pose is not None and getattr(pose, "rect", None):
            x, y, width, height = pose.rect
            if width > 24 and height > 24:
                x0 = max(0, int(x))
                y0 = max(0, int(y))
                crop = frame[y0 : y0 + int(height), x0 : x0 + int(width)]
                if crop.size > 0:
                    snapshot = bgr_to_qimage(crop)

        self._scene_canvas.set_scene(theme, code, snapshot, duration=self._scene_seconds)
        self._pause_standby_video()
        self._stack.setCurrentWidget(self._scene_canvas)
        self._scene_timer.start(int(self._scene_seconds * 1000))
        self._trigger_count += 1
        self._reset_trigger_streak()
        LOGGER.info(
            "联动触发 #%d：二维码=%s → 主题=%s（%.0f 秒后返回扫描页）",
            self._trigger_count,
            code,
            theme.title,
            self._scene_seconds,
        )

    def _finish_scene(self) -> None:
        self._scene_timer.stop()
        if self._vision_worker is not None:
            self._show_camera()
        else:
            self._show_standby()

    def selftest_summary(self) -> dict:
        """给 --selftest 使用的运行摘要，便于终端断言摄像头与联动状态。"""
        analysis = self._last_analysis
        return {
            "scanner_available": SCANNER_AVAILABLE,
            "camera_requested": self._start_camera_on_launch,
            "camera_enabled": self._camera_enabled,
            "frames": self._camera_canvas.frames,
            "triggers": self._trigger_count,
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
            self._start_camera()

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.setCursor(Qt.ArrowCursor)
            self.showNormal()
            self.resize(1280, 720)
        else:
            self.setCursor(Qt.BlankCursor)
            self.showFullScreen()

    def keyPressEvent(self, event) -> None:  # noqa: N802
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
    parser = argparse.ArgumentParser(description="外星花园 · 交互展示（阶段二：视觉检测与联动）")
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
        help="启动后立即打开摄像头进入实时扫描",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="禁用摄像头（只保留待机展示）",
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
    elif args.start_camera:
        LOGGER.info("启动后立即进入摄像头扫描（index=%s）", args.camera)

    window = StandbyWindow(
        video_path=video_path,
        volume=args.volume,
        windowed=args.windowed,
        force_fallback=args.no_video,
        camera_index=args.camera,
        camera_width=args.width,
        camera_height=args.height,
        start_camera=args.start_camera and not args.no_camera,
        camera_enabled=not args.no_camera,
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
        if args.start_camera and not args.no_camera:
            ok = summary["frames"] > 0 and not summary["camera_error"]
        print("SELFTEST_OK" if ok else "SELFTEST_FAIL")
        return 0 if ok else 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
