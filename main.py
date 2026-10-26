"""外星花园交互展示程序 —— 阶段一：待机展示主窗口。

用法：
    python main.py                    # 全屏无边框，自动查找待机视频
    python main.py --windowed         # 以 1280x720 窗口运行，便于调试
    python main.py --no-video         # 强制使用黑色呼吸文字待机页
    python main.py --video D:/standby.mp4

快捷键：
    ESC  退出程序
    F11  全屏 / 窗口切换（调试用）

待机视频默认从 assets/videos 目录查找。未找到视频或播放失败时，程序会自动
退回纯黑背景 + 呼吸文字的内置待机页，保证展厅大屏不会黑屏无提示。
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, QUrl
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QLinearGradient,
    QPainter,
    QPen,
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

BASE_DIR = Path(__file__).resolve().parent
VIDEO_DIR = BASE_DIR / "assets" / "videos"
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".webm", ".m4v"}
PREFERRED_VIDEO_KEYWORDS = ("standby", "idle", "waiting", "daiji", "待机")

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


class StandbyWindow(QWidget):
    """全屏无边框展示窗口，负责待机视频与兜底页面的切换。"""

    def __init__(
        self,
        video_path: Optional[Path] = None,
        volume: int = 0,
        windowed: bool = False,
        force_fallback: bool = False,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("外星花园 · 待机展示")
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

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._stack.setStackingMode(QStackedLayout.StackOne)

        self._idle_canvas = IdleCanvas(self)
        self._stack.addWidget(self._idle_canvas)

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
        self._stack.setCurrentWidget(self._idle_canvas)

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

    def _toggle_fullscreen(self) -> None:
        if self.isFullScreen():
            self.setCursor(Qt.ArrowCursor)
            self.showNormal()
            self.resize(1280, 720)
        else:
            self.setCursor(Qt.BlankCursor)
            self.showFullScreen()

    def keyPressEvent(self, event) -> None:  # noqa: N802
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
        LOGGER.info("正在关闭待机展示窗口")
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
    parser = argparse.ArgumentParser(description="外星花园 · 待机展示（阶段一）")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="待机视频路径；缺省时自动扫描 assets/videos",
    )
    parser.add_argument("--windowed", action="store_true", help="以 1280x720 窗口运行，便于调试")
    parser.add_argument("--no-video", action="store_true", help="强制关闭视频，使用呼吸文字待机页")
    parser.add_argument(
        "--volume",
        type=int,
        default=0,
        help="待机视频音量 0-100（默认 0，静音）",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认 INFO）",
    )
    parser.add_argument("--selftest", action="store_true", help=argparse.SUPPRESS)
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

    window = StandbyWindow(
        video_path=video_path,
        volume=args.volume,
        windowed=args.windowed,
        force_fallback=args.no_video,
    )
    window.start()

    if args.selftest:
        QTimer.singleShot(1500, app.quit)
        LOGGER.info("自检模式：1.5 秒后自动退出")

    exit_code = app.exec_()
    if args.selftest:
        print("SELFTEST_OK")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
