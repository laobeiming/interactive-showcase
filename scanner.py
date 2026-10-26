"""摄像头采集、二维码识别与图纸摆放角度校准模块（阶段二）。

本模块负责“看得见、读得出、摆得正”：
* 封装 OpenCV 摄像头的打开、读取、释放与热插拔容错；
* 封装 pyzbar 二维码识别，输出二维码内容、位置与中心点；
* 基于轮廓检测估算图纸摆放的水平倾斜角，倾斜超标时输出校正信号；
* 提供独立命令行预览 / 终端探测 / 算法自检工具，便于现场调试高拍仪。

上层（``main.py``）通过 ``CameraScanner.analyze_frame()`` 一次性获得
二维码与图纸姿态，再决定是否联动大屏。

命令行示例：
    python scanner.py                      # 默认摄像头预览
    python scanner.py --camera 1           # 指定第二路摄像头
    python scanner.py --image qr.png       # 离线识别单张图片
    python scanner.py --list-cameras       # 探测可用摄像头
    python scanner.py --probe --frames 30  # 无窗口抓取 30 帧并打印分析结果
    python scanner.py --selftest           # 合成图算法自检（不需要摄像头）
    python scanner.py --demo               # 无摄像头演示（模拟画纸 + 二维码）

预览窗口按键：q / ESC 退出，s 保存当前帧快照。
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("alien_garden.scanner")
IS_WINDOWS = platform.system() == "Windows"
PROJECT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = PROJECT_DIR / "captures"

Point = Tuple[int, int]
Quad = Tuple[Point, Point, Point, Point]
Rect = Tuple[int, int, int, int]

# 图纸水平角阈值：|倾斜角| 超过该值即认为需要校正摆放
DEFAULT_TILT_THRESHOLD_DEG = 8.0
# 图纸至少占画面面积的比例，过滤桌面纹理、二维码小方块等误检
DEFAULT_MIN_PAPER_AREA_RATIO = 0.04
# 图纸最多占画面面积的比例，避免把整幅画面边框当成图纸
DEFAULT_MAX_PAPER_AREA_RATIO = 0.92

# pyzbar 导入失败通常是缺少 libzbar 动态库；此时摄像头预览仍可使用，
# 只是二维码识别返回空结果，并通过 last_error 告知上层。
try:
    from pyzbar import pyzbar
    from pyzbar.pyzbar import ZBarSymbol

    PYZBAR_AVAILABLE = True
    PYZBAR_IMPORT_ERROR = ""
except Exception as exc:  # ImportError 或 OSError（DLL 缺失）
    pyzbar = None  # type: ignore[assignment]
    ZBarSymbol = None  # type: ignore[assignment]
    PYZBAR_AVAILABLE = False
    PYZBAR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class QRCode:
    """一条二维码识别结果。"""

    data: str
    code_type: str
    rect: Rect
    polygon: Tuple[Point, ...]

    @property
    def center(self) -> Point:
        x, y, width, height = self.rect
        return (x + width // 2, y + height // 2)

    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "type": self.code_type,
            "rect": self.rect,
            "center": self.center,
        }


@dataclass(frozen=True)
class PaperPose:
    """图纸摆放姿态（基于轮廓检测估算）。

    ``angle_deg`` 为图纸上边缘相对水平线的倾角，范围约 ``[-45, 45]``：

    * 正数：图纸右侧偏下，需要**逆时针**回正；
    * 负数：图纸右侧偏上，需要**顺时针**回正。
    """

    found: bool
    angle_deg: float = 0.0
    quad: Quad = ((0, 0), (0, 0), (0, 0), (0, 0))
    rect: Optional[Rect] = None
    area_ratio: float = 0.0
    confidence: float = 0.0
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG
    needs_correction: bool = False

    @property
    def center(self) -> Point:
        if self.rect is None:
            return (0, 0)
        x, y, width, height = self.rect
        return (x + width // 2, y + height // 2)

    @property
    def abs_angle(self) -> float:
        return abs(self.angle_deg)

    @property
    def correction_hint(self) -> str:
        """给操作者看的回正方向提示（未超标时为空字符串）。"""
        if not self.found or not self.needs_correction:
            return ""
        direction = "逆时针" if self.angle_deg > 0 else "顺时针"
        return f"请{direction}旋转图纸约 {self.abs_angle:.1f}°"

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "angle_deg": round(self.angle_deg, 2),
            "needs_correction": self.needs_correction,
            "tilt_threshold": self.tilt_threshold,
            "quad": [list(point) for point in self.quad] if self.found else [],
            "rect": self.rect,
            "center": self.center,
            "area_ratio": round(self.area_ratio, 4),
            "confidence": round(self.confidence, 4),
        }


class ScanStatus(str, Enum):
    """一帧画面的综合扫描状态。"""

    NO_PAPER = "no_paper"      # 未检测到图纸
    TILTED = "tilted"          # 检测到图纸但倾斜超标
    NO_CODE = "no_code"        # 图纸水平但未识别到二维码
    READY = "ready"            # 图纸水平且二维码识别成功，可联动


STATUS_MESSAGES = {
    ScanStatus.NO_PAPER: "未检测到完整图纸，请将画作四边完整放入扫描区",
    ScanStatus.TILTED: "图纸倾斜，请摆正后重新扫描",
    ScanStatus.NO_CODE: "未识别到二维码，请将二维码对准扫描区",
    ScanStatus.READY: "识别成功，可以联动大屏",
}


@dataclass(frozen=True)
class FrameAnalysis:
    """一帧图像的综合分析结果：二维码 + 图纸姿态 + 联动状态。"""

    status: ScanStatus
    codes: Tuple[QRCode, ...] = ()
    pose: PaperPose = PaperPose(found=False)
    frame_size: Tuple[int, int] = (0, 0)
    timestamp: float = 0.0

    @property
    def primary_code(self) -> Optional[QRCode]:
        """返回面积最大的二维码，作为当前画作的主 ID。"""
        if not self.codes:
            return None
        return max(self.codes, key=lambda code: code.rect[2] * code.rect[3])

    @property
    def code_data(self) -> Optional[str]:
        code = self.primary_code
        return code.data if code is not None else None

    @property
    def message(self) -> str:
        if self.status is ScanStatus.TILTED and self.pose.correction_hint:
            return self.pose.correction_hint
        if self.status is ScanStatus.NO_PAPER:
            if self.codes:
                return "已识别二维码，请让整张纸的四边完整进入画面"
            return "未识别到二维码，请将画作（含二维码）平放入扫描区"
        return STATUS_MESSAGES[self.status]

    @property
    def needs_correction(self) -> bool:
        return self.status is ScanStatus.TILTED

    @property
    def ready(self) -> bool:
        return self.status is ScanStatus.READY

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "message": self.message,
            "codes": [code.to_dict() for code in self.codes],
            "pose": self.pose.to_dict(),
            "frame_size": list(self.frame_size),
            "timestamp": round(self.timestamp, 3),
        }


@dataclass
class CameraConfig:
    """摄像头采集参数，后续可由配置文件覆盖。"""

    index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    warmup_frames: int = 8
    read_retries: int = 3
    open_timeout: float = 2.0


class CameraScanner:
    """摄像头 + 二维码识别 + 图纸姿态校准的统一入口。

    典型用法::

        with CameraScanner(CameraConfig(index=0)) as scanner:
            analysis = scanner.analyze_frame(scanner.read_frame())
            if analysis.ready:
                print(analysis.code_data)

    该类不依赖 Qt，可在主线程之外复用；但 OpenCV 的窗口操作需留在主线程。
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self._capture: Optional[cv2.VideoCapture] = None
        self.last_error: Optional[str] = None
        # 角度时序平滑：降低轮廓抖动造成的 HUD 数字跳动
        self._angle_history: deque[float] = deque(maxlen=5)
        self._pose_key: Optional[str] = None

    # ---- 设备管理 -----------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def _apply_capture_settings(self, capture: cv2.VideoCapture) -> None:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 尽量取最新帧，降低识别延迟

    def _warm_up(self, capture: cv2.VideoCapture) -> bool:
        """等待摄像头真正出图。

        Windows 下 DirectShow / MSMF 打开成功后，前几帧读取失败是常见现象；
        这里按时间窗口重试，避免 ``--list-cameras`` 或启动阶段误判为无设备。
        """
        deadline = time.monotonic() + max(0.2, float(self.config.open_timeout))
        good_frames = 0
        while time.monotonic() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                good_frames += 1
                if good_frames >= max(1, int(self.config.warmup_frames)):
                    return True
            else:
                time.sleep(0.03)
        return good_frames > 0

    def open(self) -> bool:
        """打开摄像头并完成预热，成功返回 True。"""
        if self.is_open:
            return True

        self.last_error = None
        if IS_WINDOWS:
            backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY)
        else:
            backends = (cv2.CAP_ANY,)

        failure_reason = "设备被占用或未连接"
        for backend in backends:
            LOGGER.debug("尝试打开摄像头 index=%s backend=%s", self.config.index, backend)
            capture = cv2.VideoCapture(self.config.index, backend)
            if not capture.isOpened():
                capture.release()
                failure_reason = f"backend={backend} 无法打开"
                continue

            try:
                self._apply_capture_settings(capture)
                if not self._warm_up(capture):
                    capture.release()
                    failure_reason = f"backend={backend} 打开后无法读取画面"
                    continue
            except Exception as exc:  # 驱动异常时继续尝试下一个后端
                capture.release()
                failure_reason = f"backend={backend} 初始化异常：{type(exc).__name__}: {exc}"
                LOGGER.debug(failure_reason)
                continue

            self._capture = capture
            actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            actual_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
            LOGGER.info(
                "摄像头已打开：index=%s，backend=%s，分辨率=%sx%s，FPS=%.1f",
                self.config.index,
                backend,
                actual_width,
                actual_height,
                actual_fps,
            )
            return True

        self.last_error = f"无法打开摄像头 index={self.config.index}（{failure_reason}）"
        LOGGER.error(self.last_error)
        return False

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧 BGR 图像；失败时返回 None 并更新 last_error。"""
        if not self.is_open and not self.open():
            return None

        assert self._capture is not None
        for attempt in range(max(1, int(self.config.read_retries))):
            ok, frame = self._capture.read()
            if ok and frame is not None:
                self.last_error = None
                return frame
            if attempt + 1 < self.config.read_retries:
                time.sleep(0.02)

        self.last_error = "摄像头读取失败（设备被占用或已断开）"
        LOGGER.warning(self.last_error)
        return None

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("摄像头已释放")

    # ---- 二维码识别 ---------------------------------------------------
    def _smooth_pose(self, pose: PaperPose, key: Optional[str]) -> PaperPose:
        """对连续帧的角度做向量平均，抑制 HUD 抖动与阈值边缘反复跳变。"""
        if not pose.found:
            self._angle_history.clear()
            self._pose_key = None
            return pose

        if key != self._pose_key:
            self._angle_history.clear()
            self._pose_key = key
        self._angle_history.append(pose.angle_deg)
        if len(self._angle_history) < 3:
            return pose

        # 角度在 ±45° 处会回绕，使用 2θ 向量平均避免 44° 与 -44° 被判成 0°
        sin_sum = sum(math.sin(math.radians(2.0 * angle)) for angle in self._angle_history)
        cos_sum = sum(math.cos(math.radians(2.0 * angle)) for angle in self._angle_history)
        smoothed = _normalize_angle(math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0)
        return replace(
            pose,
            angle_deg=smoothed,
            needs_correction=abs(smoothed) > float(pose.tilt_threshold),
        )

    def scan_frame(
        self,
        frame: np.ndarray,
        roi: Optional[Rect] = None,
    ) -> List[QRCode]:
        """识别 BGR 图像中的二维码。

        Args:
            frame: OpenCV BGR 图像。
            roi: 可选的识别区域 ``(x, y, w, h)``，用于只扫描图纸区域、降低误识别。
        """
        if frame is None:
            self.last_error = "待识别图像为空"
            return []
        if not PYZBAR_AVAILABLE:
            self.last_error = f"pyzbar 不可用：{PYZBAR_IMPORT_ERROR}"
            LOGGER.warning(self.last_error)
            return []

        target = frame
        offset_x = 0
        offset_y = 0
        if roi is not None:
            frame_height, frame_width = frame.shape[:2]
            x, y, width, height = (int(value) for value in roi)
            x = max(0, min(x, frame_width - 1))
            y = max(0, min(y, frame_height - 1))
            width = max(0, min(width, frame_width - x))
            height = max(0, min(height, frame_height - y))
            if width <= 0 or height <= 0:
                self.last_error = f"无效的 ROI：{roi}"
                return []
            target = frame[y : y + height, x : x + width]
            offset_x, offset_y = x, y

        try:
            gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            decoded_items = pyzbar.decode(gray, symbols=[ZBarSymbol.QRCODE])
        except Exception as exc:  # 解码异常不能中断主流程
            self.last_error = f"二维码解码失败：{type(exc).__name__}: {exc}"
            LOGGER.exception(self.last_error)
            return []

        results: List[QRCode] = []
        for item in decoded_items:
            text = item.data.decode("utf-8", errors="replace")
            rect = (
                int(item.rect.left) + offset_x,
                int(item.rect.top) + offset_y,
                int(item.rect.width),
                int(item.rect.height),
            )
            polygon = tuple(
                (int(point.x) + offset_x, int(point.y) + offset_y)
                for point in item.polygon
            )
            results.append(
                QRCode(
                    data=text,
                    code_type=item.type,
                    rect=rect,
                    polygon=polygon,
                )
            )

        if results:
            self.last_error = None
        return results

    def analyze_frame(
        self,
        frame: np.ndarray,
        *,
        roi: Optional[Rect] = None,
        tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
        min_paper_area_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
        detect_pose: bool = True,
    ) -> FrameAnalysis:
        """一次性完成二维码识别 + 图纸水平角校准，返回联动状态。

        Args:
            frame: OpenCV BGR 图像。
            roi: 可选的二维码识别区域 ``(x, y, w, h)``。
            tilt_threshold: 允许的最大水平倾斜角（度）。
            min_paper_area_ratio: 图纸最小面积占比，过滤小面积误检。
            detect_pose: 关闭后只做二维码识别（兼容旧流程 / 性能调试）。

        Returns:
            ``FrameAnalysis``；``status`` 为 ``READY`` 时上层可安全触发联动。
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            self.last_error = "待分析图像为空"
            return FrameAnalysis(status=ScanStatus.NO_PAPER, timestamp=time.time())

        frame_height, frame_width = frame.shape[:2]
        codes = self.scan_frame(frame, roi=roi)

        if detect_pose:
            primary = (
                max(codes, key=lambda code: code.rect[2] * code.rect[3]) if codes else None
            )
            if primary is None:
                # 没有二维码时不猜纸张轮廓：宁可提示“未识别二维码”，
                # 也不要把天花板、窗户、墙面等亮区误框成图纸。
                pose = PaperPose(found=False, tilt_threshold=tilt_threshold)
                self._angle_history.clear()
                self._pose_key = None
                status = ScanStatus.NO_PAPER
            else:
                pose = detect_paper_pose(
                    frame,
                    tilt_threshold=tilt_threshold,
                    min_area_ratio=min_paper_area_ratio,
                    hint_point=primary.center,
                    anchor_polygon=primary.polygon,
                    anchor_rect=primary.rect,
                )
                pose = self._smooth_pose(pose, primary.data)
                if not pose.found:
                    status = ScanStatus.NO_PAPER
                elif pose.needs_correction:
                    status = ScanStatus.TILTED
                else:
                    status = ScanStatus.READY
        else:
            pose = PaperPose(found=False, tilt_threshold=tilt_threshold)
            status = ScanStatus.READY if codes else ScanStatus.NO_CODE

        return FrameAnalysis(
            status=status,
            codes=tuple(codes),
            pose=pose,
            frame_size=(frame_width, frame_height),
            timestamp=time.time(),
        )

    def scan_once(self, roi: Optional[Rect] = None) -> Tuple[Optional[np.ndarray], List[QRCode]]:
        """抓取一帧并立即识别，返回 ``(frame, codes)``。"""
        frame = self.read_frame()
        if frame is None:
            return None, []
        return frame, self.scan_frame(frame, roi=roi)

    # ---- 上下文管理 ---------------------------------------------------
    def __enter__(self) -> "CameraScanner":
        self.open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()


def _normalize_angle(deg: float) -> float:
    """把任意直线角度归一化到 ``(-45, 45]``，用于求“相对水平/垂直”的偏差。"""
    while deg <= -45.0:
        deg += 90.0
    while deg > 45.0:
        deg -= 90.0
    return deg


def _order_quad(points: np.ndarray) -> Quad:
    """把任意顺序的四个角点整理为 左上、右上、右下、左下。"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if pts.shape[0] != 4:
        raise ValueError("需要恰好四个角点")

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered[0] = pts[int(np.argmin(sums))]  # 左上：x+y 最小
    ordered[2] = pts[int(np.argmax(sums))]  # 右下：x+y 最大
    ordered[1] = pts[int(np.argmin(diffs))]  # 右上：y-x 最小
    ordered[3] = pts[int(np.argmax(diffs))]  # 左下：y-x 最大

    # 极端旋转时上述规则可能得到重复点，退回按极角排序
    if len({tuple(point) for point in ordered}) < 4:
        center = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
        pts = pts[np.argsort(angles)]
        start = int(np.argmin(pts.sum(axis=1)))
        pts = np.roll(pts, -start, axis=0)
        ordered = pts

    return (
        (int(round(float(ordered[0][0]))), int(round(float(ordered[0][1])))),
        (int(round(float(ordered[1][0]))), int(round(float(ordered[1][1])))),
        (int(round(float(ordered[2][0]))), int(round(float(ordered[2][1])))),
        (int(round(float(ordered[3][0]))), int(round(float(ordered[3][1])))),
    )


def _quad_tilt_deg(quad: Quad) -> float:
    """按四条边的加权平均，计算图纸相对水平线的倾斜角。"""
    tl, tr, br, bl = (np.array(point, dtype=np.float64) for point in quad)
    edges = ((tl, tr), (bl, br), (tl, bl), (tr, br))
    total = 0.0
    weight_sum = 0.0
    for start, end in edges:
        delta = end - start
        length = float(np.hypot(delta[0], delta[1]))
        if length < 8.0:  # 太短的边受噪声影响大，直接忽略
            continue
        raw_angle = math.degrees(math.atan2(float(delta[1]), float(delta[0])))
        total += _normalize_angle(raw_angle) * length
        weight_sum += length
    if weight_sum <= 0.0:
        return 0.0
    return _normalize_angle(total / weight_sum)


def _quad_corner_angles(quad: Quad) -> List[float]:
    """返回四边形的四个内角（度），用于剔除三角形、细长条等伪四边形。"""
    points = np.asarray(quad, dtype=np.float64)
    angles: List[float] = []
    for index in range(4):
        previous = points[(index - 1) % 4]
        current = points[index]
        following = points[(index + 1) % 4]
        first = previous - current
        second = following - current
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        if denominator <= 1e-6:
            continue
        cosine = float(np.dot(first, second) / denominator)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def _quad_contains_point(quad: Quad, point: Point) -> bool:
    contour = np.asarray(quad, dtype=np.float32).reshape(-1, 1, 2)
    return (
        cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), False)
        >= 0
    )


def _anchor_inside(
    quad: Quad,
    hint_point: Point,
    anchor_polygon: Optional[Sequence[Point]],
    anchor_rect: Optional[Rect],
    quad_area: float,
) -> bool:
    """二维码必须是候选纸张的一部分，避免把天花板 / 窗户当成图纸。"""
    if not _quad_contains_point(quad, hint_point):
        return False

    if anchor_polygon:
        inside = sum(1 for point in anchor_polygon if _quad_contains_point(quad, point))
        if inside < max(3, len(anchor_polygon) - 1):
            return False

    if anchor_rect is not None:
        qr_area = max(1.0, float(anchor_rect[2]) * float(anchor_rect[3]))
        # 纸张面积至少是二维码面积的 2 倍，排除只框住二维码的小四边形
        if quad_area < qr_area * 2.0:
            return False

    return True


def _quad_from_contour(
    contour: np.ndarray,
    box: np.ndarray,
) -> Tuple[Quad, float]:
    """优先用 approxPolyDP 贴合真实纸张边界，失败时退回最小外接矩形。"""
    perimeter = cv2.arcLength(contour, True)
    if perimeter > 0:
        for ratio in (0.02, 0.03, 0.045, 0.06):
            approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                quad = _order_quad(approx.reshape(-1, 2))
                quad_area = abs(
                    float(cv2.contourArea(np.asarray(quad, dtype=np.float32)))
                )
                if quad_area > 0:
                    return quad, quad_area

    quad = _order_quad(box)
    quad_area = abs(float(cv2.contourArea(box.astype(np.float32))))
    return quad, quad_area


def _paper_appearance(
    gray: np.ndarray,
    hsv: Optional[np.ndarray],
    edges: np.ndarray,
    quad: Quad,
) -> Tuple[float, float, float, float]:
    """统计候选区域的外观特征：(亮度, 饱和度, 灰度标准差, 内部边缘密度)。"""
    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillConvexPoly(
        mask,
        np.asarray(quad, dtype=np.int32).reshape(-1, 1, 2),
        255,
    )
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
    if cv2.countNonZero(inner) < 200:
        inner = mask

    pixel_count = float(cv2.countNonZero(inner))
    if pixel_count <= 0:
        return 0.0, 255.0, 255.0, 1.0

    selected = inner > 0
    gray_values = gray[selected]
    mean_gray = float(gray_values.mean())
    std_gray = float(gray_values.std())
    edge_density = float(
        cv2.countNonZero(cv2.bitwise_and(edges, inner)) / pixel_count
    )

    if hsv is not None:
        saturation = float(hsv[..., 1][selected].mean())
        value = float(hsv[..., 2][selected].mean())
    else:
        saturation = 0.0
        value = mean_gray

    return value, saturation, std_gray, edge_density


def _paper_masks(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """生成两套候选掩膜：边缘轮廓 + Otsu 亮纸区域。

    高拍仪下白纸与深色桌面对比明显，Otsu 很稳；当纸张颜色接近桌面时，
    边缘轮廓可以补足。两套掩膜都交给 ``findContours``，取评分最高者。
    """
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    median = float(np.median(blurred))
    lower = int(max(0.0, 0.66 * median))
    upper = int(min(255.0, max(lower + 30.0, 1.33 * median)))
    edges = cv2.Canny(blurred, lower, upper)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

    _, bright = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    return edges, bright


def detect_paper_pose(
    frame: np.ndarray,
    *,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
    min_area_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
    max_area_ratio: float = DEFAULT_MAX_PAPER_AREA_RATIO,
    hint_point: Optional[Point] = None,
    anchor_polygon: Optional[Sequence[Point]] = None,
    anchor_rect: Optional[Rect] = None,
) -> PaperPose:
    """基于轮廓检测估算图纸摆放的水平角。

    处理流程：
    1. 灰度化 + 高斯模糊，生成 Canny 边缘掩膜与 Otsu 亮区掩膜；
    2. ``findContours`` 提取外轮廓，用 ``approxPolyDP`` / ``minAreaRect`` 拟合四边形；
    3. 按面积、矩形度、长宽比、凸性、四角角度、白纸外观筛选候选；
    4. 有二维码时要求二维码落在纸张内部（锚定），彻底排除天花板、窗户等大亮块；
    5. 由四条边角度加权平均得到倾斜角，超过阈值时 ``needs_correction=True``。

    Args:
        frame: OpenCV BGR / 灰度图像。
        tilt_threshold: 允许的最大倾斜角（度），默认 8°。
        min_area_ratio: 图纸最小面积占比。
        max_area_ratio: 图纸最大面积占比。
        hint_point: 二维码中心点。提供时只接受包含该点的候选，用于锚定真实纸张。
        anchor_polygon: 二维码四个角点，提供时要求角点也落在纸张内部。
        anchor_rect: 二维码外接矩形，用于排除“只框住二维码”的小四边形。

    Returns:
        ``PaperPose``；未找到完整图纸时 ``found=False``。
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return PaperPose(found=False, tilt_threshold=tilt_threshold)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if frame.ndim == 3 else None
    frame_height, frame_width = gray.shape[:2]
    frame_area = float(frame_height * frame_width)
    if frame_area <= 0:
        return PaperPose(found=False, tilt_threshold=tilt_threshold)
    min_area = frame_area * max(0.001, float(min_area_ratio))
    anchored = hint_point is not None

    edges, bright = _paper_masks(gray)

    best_quad: Optional[Quad] = None
    best_area_ratio = 0.0
    best_confidence = 0.0
    best_score = -1.0

    for mask in (edges, bright):
        raw_contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        contours = sorted(raw_contours, key=cv2.contourArea, reverse=True)[:12]
        for contour in contours:
            contour_area = abs(float(cv2.contourArea(contour)))
            if contour_area < min_area * 0.8:
                continue

            rect_fit = cv2.minAreaRect(contour)
            box_width, box_height = rect_fit[1]
            if min(box_width, box_height) < 40.0:
                continue

            box = cv2.boxPoints(rect_fit)
            box_area = abs(float(cv2.contourArea(box.astype(np.float32))))
            if box_area <= 0:
                continue

            area_ratio = box_area / frame_area
            if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                continue

            rectangularity = contour_area / box_area
            if rectangularity < (0.66 if anchored else 0.70):
                continue

            aspect = min(box_width, box_height) / max(box_width, box_height)
            if anchored:
                if aspect < 0.42 or aspect > 2.10:
                    continue
            elif aspect < 0.50 or aspect > 1.85:
                continue

            try:
                quad, quad_area = _quad_from_contour(contour, box)
            except (ValueError, cv2.error):
                continue
            if quad_area <= 0.0 or quad_area / frame_area > max_area_ratio:
                continue

            corners = _quad_corner_angles(quad)
            if len(corners) != 4 or min(corners) < 52.0 or max(corners) > 128.0:
                continue

            xs = [point[0] for point in quad]
            ys = [point[1] for point in quad]
            # 四角被画面裁掉时无法还原真实角度，直接放弃该候选
            if (
                min(xs) <= 3
                or min(ys) <= 3
                or max(xs) >= frame_width - 4
                or max(ys) >= frame_height - 4
            ):
                continue

            # 有二维码时，纸张必须包住二维码；没有二维码时才走严格的白纸外观筛选
            if anchored:
                assert hint_point is not None
                if not _anchor_inside(
                    quad,
                    hint_point,
                    anchor_polygon,
                    anchor_rect,
                    quad_area,
                ):
                    continue

            value, saturation, gray_std, edge_density = _paper_appearance(
                gray, hsv, edges, quad
            )
            if anchored:
                if value < 85.0 or saturation > 140.0:
                    continue
            else:
                center_y_ratio = (min(ys) + max(ys)) / 2.0 / frame_height
                if center_y_ratio < 0.25:  # 画面上方多为天花板 / 灯带，直接排除
                    continue
                if value < 110.0 or saturation > 100.0:
                    continue
                # 儿童画本身包含线条与二维码，允许一定的灰度波动和边缘密度
                if gray_std > 85.0 or edge_density > 0.30:
                    continue

            aspect_score = 1.0 - min(1.0, abs(aspect - 0.707) / 0.707)
            whiteness = min(1.0, max(0.0, (value - 80.0) / 120.0))
            uniformity = max(0.0, 1.0 - gray_std / 80.0)
            cleanness = max(0.0, 1.0 - edge_density / 0.30)
            appearance = 0.50 * whiteness + 0.25 * uniformity + 0.25 * cleanness

            center_x = (min(xs) + max(xs)) / 2.0 / frame_width
            center_y = (min(ys) + max(ys)) / 2.0 / frame_height
            centrality = max(
                0.0,
                1.0 - math.hypot(center_x - 0.5, center_y - 0.5) * 1.6,
            )
            score = (
                area_ratio
                * (0.55 + 0.45 * rectangularity)
                * (0.70 + 0.30 * aspect_score)
                * (0.90 + 0.10 * centrality)
                * (0.65 + 0.35 * appearance)
            )
            score *= 1.55 if anchored else 0.90

            if score > best_score:
                best_score = score
                best_quad = quad
                best_area_ratio = quad_area / frame_area
                best_confidence = min(1.0, 0.35 + 0.65 * appearance)

    if best_quad is None:
        return PaperPose(found=False, tilt_threshold=tilt_threshold)

    tilt_deg = _quad_tilt_deg(best_quad)
    xs = [point[0] for point in best_quad]
    ys = [point[1] for point in best_quad]
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(frame_width, max(xs))
    y1 = min(frame_height, max(ys))
    rect: Rect = (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    return PaperPose(
        found=True,
        angle_deg=tilt_deg,
        quad=best_quad,
        rect=rect,
        area_ratio=best_area_ratio,
        confidence=best_confidence,
        tilt_threshold=tilt_threshold,
        needs_correction=abs(tilt_deg) > float(tilt_threshold),
    )


def _ascii_label(text: str, limit: int = 48) -> str:
    """OpenCV 的 putText 只支持 ASCII，预览窗口里把非 ASCII 字符替换掉。"""
    safe = text.encode("ascii", errors="replace").decode("ascii")
    return safe if len(safe) <= limit else safe[: limit - 3] + "..."


def _put_text(
    canvas: np.ndarray,
    text: str,
    position: Point,
    color: Tuple[int, int, int],
    *,
    scale: float = 0.65,
    thickness: int = 2,
) -> None:
    """带黑色描边的 OpenCV 文本，保证复杂背景下也能看清。"""
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


_STATUS_ASCII = {
    ScanStatus.NO_PAPER: "STATUS: NO PAPER",
    ScanStatus.TILTED: "STATUS: TILTED - LEVEL THE PAPER",
    ScanStatus.NO_CODE: "STATUS: PAPER OK - NO QR CODE",
    ScanStatus.READY: "STATUS: READY - LINK NOW",
}

_STATUS_COLORS = {
    ScanStatus.NO_PAPER: (205, 205, 205),
    ScanStatus.TILTED: (60, 185, 255),
    ScanStatus.NO_CODE: (80, 215, 255),
    ScanStatus.READY: (90, 255, 175),
}


def annotate_frame(
    frame: np.ndarray,
    codes: Sequence[QRCode],
    *,
    show_hint: bool = True,
    pose: Optional[PaperPose] = None,
    analysis: Optional[FrameAnalysis] = None,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
) -> np.ndarray:
    """在图像副本上框出图纸与二维码，返回用于预览的 BGR 图像。"""
    canvas = frame.copy()
    if analysis is not None:
        pose = analysis.pose
        tilt_threshold = analysis.pose.tilt_threshold

    # 图纸轮廓：水平为绿色，倾斜超标为橙色
    if pose is not None and pose.found:
        paper_ok = not pose.needs_correction
        paper_color = (90, 255, 175) if paper_ok else (60, 185, 255)
        points = np.array(pose.quad, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], True, paper_color, 2, cv2.LINE_AA)
        cv2.drawMarker(
            canvas,
            pose.center,
            paper_color,
            markerType=cv2.MARKER_CROSS,
            markerSize=22,
            thickness=2,
        )
        paper_label = f"PAPER {pose.angle_deg:+.1f} deg (limit {tilt_threshold:.1f})"
        label_y = max(20, min(pose.quad, key=lambda point: point[1])[1] - 10)
        _put_text(canvas, paper_label, (max(8, pose.rect[0] if pose.rect else 8), label_y), paper_color)

    # 二维码框与内容
    for code in codes:
        x, y, width, height = code.rect
        if len(code.polygon) >= 4:
            points = np.array(code.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], True, (0, 255, 170), 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 220, 150), 2)
        cv2.circle(canvas, code.center, 5, (0, 255, 200), -1, cv2.LINE_AA)

        label = f"QR: {_ascii_label(code.data)}"
        label_y = max(24, y - 12)
        _put_text(canvas, label, (x, label_y), (0, 255, 170), scale=0.7)

    # 左上角综合状态
    if analysis is not None:
        status_color = _STATUS_COLORS[analysis.status]
        _put_text(canvas, _STATUS_ASCII[analysis.status], (16, 34), status_color, scale=0.72)
        if pose is not None and pose.found:
            detail = f"tilt {pose.angle_deg:+.1f} / {tilt_threshold:.1f} deg   area {pose.area_ratio:.1%}"
        else:
            detail = "paper contour not found"
        _put_text(canvas, detail, (16, 62), status_color, scale=0.55, thickness=1)

    if show_hint:
        hint = "q / ESC: quit    s: snapshot"
        height = int(canvas.shape[0])
        _put_text(canvas, hint, (16, height - 16), (235, 245, 240), scale=0.6, thickness=1)
    return canvas


def _render_demo_qr(text: str, size: int) -> np.ndarray:
    """生成带白色静区的二维码图片，用于无摄像头演示。"""
    encoder = cv2.QRCodeEncoder_create()
    modules = encoder.encode(text)
    qr_image = cv2.resize(modules, (size, size), interpolation=cv2.INTER_NEAREST)
    if qr_image.ndim == 2:
        qr_image = cv2.cvtColor(qr_image, cv2.COLOR_GRAY2BGR)
    border = max(10, size // 12)
    return cv2.copyMakeBorder(
        qr_image,
        border,
        border,
        border,
        border,
        cv2.BORDER_CONSTANT,
        value=255,
    )


def build_demo_frame(
    width: int = 1280,
    height: int = 720,
    elapsed: float = 0.0,
    qr_text: str = "ALIEN-GARDEN-DEMO-001",
) -> np.ndarray:
    """生成一帧“桌面上的儿童画纸”模拟画面，无需真实摄像头。

    画面包含：深色桌面、手绘外星人、二维码、扫描框和轻微摆动，
    可直接交给 ``CameraScanner.scan_frame()`` 做识别演示。
    """
    width = max(960, int(width))
    height = max(540, int(height))

    # 深色桌面渐变 + 轻微噪点
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None, None]
    top = np.array([30.0, 27.0, 24.0], dtype=np.float32)
    bottom = np.array([8.0, 10.0, 10.0], dtype=np.float32)
    frame = np.broadcast_to(
        top * (1.0 - ramp) + bottom * ramp,
        (height, width, 3),
    ).copy()
    noise = np.random.default_rng(7).normal(0.0, 1.6, (height, width, 1))
    frame = np.clip(frame + noise.astype(np.float32), 0, 255).astype(np.uint8)

    # 画纸
    paper_height = int(min(height * 0.84, width * 0.72))
    paper_width = int(paper_height / 1.42)
    paper = np.full((paper_height, paper_width, 3), (238, 241, 235), dtype=np.uint8)
    for y in range(36, paper_height, 30):
        cv2.line(paper, (0, y), (paper_width, y), (229, 233, 226), 1)
    margin_x = int(paper_width * 0.06)
    cv2.line(paper, (margin_x, 0), (margin_x, paper_height), (214, 224, 230), 2)
    cv2.rectangle(
        paper,
        (2, 2),
        (paper_width - 3, paper_height - 3),
        (205, 210, 202),
        2,
    )

    # 手绘外星人
    head_center = (int(paper_width * 0.34), int(paper_height * 0.27))
    head_axes = (int(paper_width * 0.15), int(paper_height * 0.09))
    cv2.ellipse(paper, head_center, head_axes, 0, 0, 360, (66, 158, 92), 3, cv2.LINE_AA)
    eye_offset_x = head_axes[0] // 2
    eye_y = head_center[1] - head_axes[1] // 4
    for sign in (-1, 1):
        eye = (head_center[0] + sign * eye_offset_x, eye_y)
        cv2.circle(paper, eye, max(4, paper_width // 70), (38, 45, 45), -1, cv2.LINE_AA)
    for sign in (-1, 1):
        start = (
            head_center[0] + sign * head_axes[0] // 2,
            head_center[1] - head_axes[1] + 6,
        )
        end = (
            start[0] + sign * int(paper_width * 0.05),
            start[1] - int(paper_height * 0.06),
        )
        cv2.line(paper, start, end, (66, 158, 92), 2, cv2.LINE_AA)
        cv2.circle(paper, end, max(5, paper_width // 65), (60, 200, 245), -1, cv2.LINE_AA)

    body_center = (head_center[0], int(paper_height * 0.52))
    body_axes = (int(paper_width * 0.12), int(paper_height * 0.14))
    cv2.ellipse(paper, body_center, body_axes, 0, 0, 360, (86, 178, 112), 3, cv2.LINE_AA)
    arm_y = body_center[1] - int(paper_height * 0.03)
    cv2.line(
        paper,
        (body_center[0] - body_axes[0], arm_y),
        (body_center[0] - body_axes[0] - int(paper_width * 0.12), arm_y - int(paper_height * 0.05)),
        (86, 178, 112),
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        paper,
        (body_center[0] + body_axes[0], arm_y),
        (body_center[0] + body_axes[0] + int(paper_width * 0.12), arm_y - int(paper_height * 0.05)),
        (86, 178, 112),
        3,
        cv2.LINE_AA,
    )
    leg_top = body_center[1] + body_axes[1]
    for sign in (-1, 1):
        cv2.line(
            paper,
            (body_center[0] + sign * body_axes[0] // 2, leg_top),
            (body_center[0] + sign * int(paper_width * 0.08), leg_top + int(paper_height * 0.09)),
            (86, 178, 112),
            3,
            cv2.LINE_AA,
        )

    # 彩色星星
    stars = (
        (0.18, 0.10, (60, 200, 245)),
        (0.12, 0.40, (200, 120, 230)),
        (0.40, 0.20, (80, 180, 250)),
        (0.30, 0.62, (90, 210, 120)),
    )
    for x_ratio, y_ratio, color in stars:
        cv2.drawMarker(
            paper,
            (int(paper_width * x_ratio), int(paper_height * y_ratio)),
            color,
            cv2.MARKER_STAR,
            max(12, paper_width // 28),
            2,
            cv2.LINE_AA,
        )

    # 二维码固定贴在画纸右上角
    qr_core = int(paper_width * 0.38)
    qr_image = _render_demo_qr(qr_text, qr_core)
    qr_height, qr_width = qr_image.shape[:2]
    qr_x = paper_width - qr_width - int(paper_width * 0.055)
    qr_y = int(paper_height * 0.075)
    paper[qr_y : qr_y + qr_height, qr_x : qr_x + qr_width] = qr_image
    cv2.putText(
        paper,
        "SCAN ME",
        (qr_x, qr_y + qr_height + int(paper_height * 0.035)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.45, paper_width / 650),
        (70, 70, 70),
        2,
        cv2.LINE_AA,
    )

    # 底部彩色蜡笔线 + 标题
    for index, color in enumerate(((90, 90, 230), (80, 180, 240), (80, 200, 120), (220, 150, 60))):
        y = int(paper_height * 0.78) + index * max(5, paper_height // 80)
        cv2.line(
            paper,
            (int(paper_width * 0.52), y),
            (int(paper_width * 0.92), y + int(paper_height * 0.01)),
            color,
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        paper,
        "MY ALIEN",
        (int(paper_width * 0.08), int(paper_height * 0.93)),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        max(0.7, paper_width / 500),
        (170, 80, 45),
        2,
        cv2.LINE_AA,
    )

    # 轻微旋转模拟手持摆放角度，同时保持二维码可识别
    angle = math.sin(elapsed * 0.8) * 2.2
    matrix = cv2.getRotationMatrix2D(
        (paper_width / 2.0, paper_height / 2.0),
        angle,
        1.0,
    )
    rotated = cv2.warpAffine(
        paper,
        matrix,
        (paper_width, paper_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(18, 18, 16),
    )

    # 画纸在桌面上的轻微漂移
    x = (width - paper_width) // 2 + int(math.sin(elapsed * 0.6) * width * 0.012)
    y = (height - paper_height) // 2 + int(math.cos(elapsed * 0.45) * height * 0.015)
    x = max(0, min(width - paper_width, x))
    y = max(0, min(height - paper_height, y))
    frame[y : y + paper_height, x : x + paper_width] = rotated

    # 扫描取景框与演示标识
    pad = 18
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(width - 1, x + paper_width + pad)
    y1 = min(height - 1, y + paper_height + pad)
    corner_length = max(20, int(min(width, height) * 0.04))
    corner_color = (90, 230, 180)
    for cx, cy, dx, dy in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * corner_length, cy), corner_color, 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner_length), corner_color, 2, cv2.LINE_AA)
    demo_label = "DEMO MODE - simulated camera (no hardware)"
    cv2.putText(
        frame,
        demo_label,
        (16, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        demo_label,
        (16, height - 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        corner_color,
        1,
        cv2.LINE_AA,
    )
    return frame


def _grab_first_frame(capture: cv2.VideoCapture, timeout: float) -> Optional[np.ndarray]:
    """在时间窗口内等待摄像头出图，用于设备探测与预热。"""
    deadline = time.monotonic() + max(0.2, float(timeout))
    while time.monotonic() < deadline:
        ok, frame = capture.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.03)
    return None


def find_available_cameras(max_index: int = 5) -> List[Tuple[int, int, int]]:
    """探测前 ``max_index`` 路摄像头，返回 ``[(index, width, height), ...]``。

    Windows 摄像头打开后往往需要等待几百毫秒才出图，这里不再“读一次就判死”，
    而是按后端回退并在 1.5 秒窗口内等待首帧。
    """
    available: List[Tuple[int, int, int]] = []
    backends = (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY) if IS_WINDOWS else (cv2.CAP_ANY,)

    for index in range(max_index + 1):
        for backend in backends:
            capture = cv2.VideoCapture(index, backend)
            if not capture.isOpened():
                capture.release()
                continue

            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            frame = _grab_first_frame(capture, 1.5)
            capture.release()
            if frame is not None:
                height, width = frame.shape[:2]
                available.append((index, width, height))
                break
    return available


def _decode_image(
    image_path: Path,
    *,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
    min_paper_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
    detect_pose: bool = True,
) -> int:
    """离线识别单张图片，并把标注结果保存到 captures/。"""
    if not PYZBAR_AVAILABLE:
        LOGGER.error("pyzbar 不可用：%s", PYZBAR_IMPORT_ERROR)
        return 2

    frame = _imread_unicode(image_path)
    if frame is None:
        LOGGER.error("无法读取图片：%s", image_path)
        return 2

    scanner = CameraScanner()
    analysis = scanner.analyze_frame(
        frame,
        tilt_threshold=tilt_threshold,
        min_paper_area_ratio=min_paper_ratio,
        detect_pose=detect_pose,
    )
    codes = list(analysis.codes)
    annotated = annotate_frame(frame, codes, show_hint=False, analysis=analysis)

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CAPTURE_DIR / f"annotated_{image_path.stem}_{int(time.time())}.jpg"
    if not _imwrite_unicode(output_path, annotated):
        print(f"标注结果保存失败：{output_path}")

    print(f"[STATUS] {analysis.status.value} | {analysis.message}")
    if analysis.pose.found:
        print(
            f"[PAPER] angle={analysis.pose.angle_deg:+.1f} deg "
            f"limit={tilt_threshold:.1f} area={analysis.pose.area_ratio:.1%} "
            f"needs_correction={analysis.pose.needs_correction}"
        )
    else:
        print("[PAPER] 未检测到图纸轮廓（仍会尝试识别画面中的二维码）")
    for code in codes:
        print(f"[QR] {code.data}  rect={code.rect}  center={code.center}")
    print(f"识别到 {len(codes)} 个二维码，标注结果已保存：{output_path}")
    if not codes and scanner.last_error:
        print(f"提示：{scanner.last_error}")
    return 0 if codes else 1


def _preview(
    camera_index: int,
    width: int,
    height: int,
    *,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
    min_paper_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
    detect_pose: bool = True,
) -> int:
    """打开摄像头实时预览，叠加二维码与图纸倾斜角。"""
    scanner = CameraScanner(CameraConfig(index=camera_index, width=width, height=height))
    if not scanner.open():
        print(f"无法打开摄像头 {camera_index}。请检查连接或改用 --list-cameras 探测。")
        return 1
    if not PYZBAR_AVAILABLE:
        LOGGER.warning("pyzbar 不可用（%s），仅显示摄像头画面", PYZBAR_IMPORT_ERROR)

    window_name = "Alien Garden - Scanner Preview (q/ESC)"
    last_seen: dict[str, float] = {}
    last_status: Optional[ScanStatus] = None
    last_status_log = 0.0
    read_failures = 0
    try:
        while True:
            frame = scanner.read_frame()
            if frame is None:
                read_failures += 1
                if read_failures >= 30:
                    LOGGER.error("连续读取失败，结束预览")
                    break
                time.sleep(0.1)
                continue
            read_failures = 0

            analysis = scanner.analyze_frame(
                frame,
                tilt_threshold=tilt_threshold,
                min_paper_area_ratio=min_paper_ratio,
                detect_pose=detect_pose,
            )
            now = time.monotonic()
            for code in analysis.codes:
                if now - last_seen.get(code.data, 0.0) > 2.0:
                    last_seen[code.data] = now
                    LOGGER.info("识别到二维码：%s（中心点 %s）", code.data, code.center)

            if analysis.status is not last_status or now - last_status_log > 3.0:
                if analysis.pose.found:
                    LOGGER.info(
                        "状态=%s 倾斜=%+.1f°（阈值 %.1f°）二维码=%s",
                        analysis.status.value,
                        analysis.pose.angle_deg,
                        tilt_threshold,
                        [code.data for code in analysis.codes],
                    )
                else:
                    LOGGER.info(
                        "状态=%s 未检测到图纸轮廓 二维码=%s",
                        analysis.status.value,
                        [code.data for code in analysis.codes],
                    )
                last_status = analysis.status
                last_status_log = now

            cv2.imshow(window_name, annotate_frame(frame, analysis.codes, analysis=analysis))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                output_path = CAPTURE_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                if _imwrite_unicode(output_path, frame):
                    LOGGER.info("快照已保存：%s", output_path)
    except KeyboardInterrupt:
        LOGGER.info("用户中断预览")
    finally:
        scanner.release()
        cv2.destroyAllWindows()
    return 0


def _demo(
    width: int,
    height: int,
    *,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
    min_paper_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
    detect_pose: bool = True,
) -> int:
    """无摄像头演示模式：动态生成模拟画纸并实时识别二维码。"""
    if not PYZBAR_AVAILABLE:
        LOGGER.warning("pyzbar 不可用（%s），演示画面会正常显示但不会识别", PYZBAR_IMPORT_ERROR)

    scanner = CameraScanner()
    window_name = "Alien Garden - Demo Scanner (q/ESC)"
    last_seen: dict[str, float] = {}
    started = time.monotonic()
    LOGGER.info("无摄像头演示模式已启动，按 q 或 ESC 退出")
    try:
        while True:
            elapsed = time.monotonic() - started
            frame = build_demo_frame(width, height, elapsed)
            analysis = scanner.analyze_frame(
                frame,
                tilt_threshold=tilt_threshold,
                min_paper_area_ratio=min_paper_ratio,
                detect_pose=detect_pose,
            )

            now = time.monotonic()
            for code in analysis.codes:
                if now - last_seen.get(code.data, 0.0) > 2.0:
                    last_seen[code.data] = now
                    LOGGER.info("识别到演示二维码：%s（中心点 %s）", code.data, code.center)

            cv2.imshow(window_name, annotate_frame(frame, analysis.codes, analysis=analysis))
            key = cv2.waitKey(33) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                output_path = CAPTURE_DIR / f"demo_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                if _imwrite_unicode(output_path, frame):
                    LOGGER.info("演示快照已保存：%s", output_path)
    except KeyboardInterrupt:
        LOGGER.info("用户中断演示")
    finally:
        scanner.release()
        cv2.destroyAllWindows()
    return 0


def _imread_unicode(image_path: Path) -> Optional[np.ndarray]:
    """读取图片，兼容 Windows 中文 / 非 ASCII 路径。"""
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception as exc:
        LOGGER.error("读取图片失败：%s（%s）", image_path, exc)
        return None


def _imwrite_unicode(output_path: Path, image: np.ndarray) -> bool:
    """保存图片，兼容 Windows 中文 / 非 ASCII 路径。"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = output_path.suffix or ".jpg"
        ok, buffer = cv2.imencode(suffix, image)
        if not ok:
            LOGGER.error("图片编码失败：%s", output_path)
            return False
        buffer.tofile(str(output_path))
        return True
    except Exception as exc:
        LOGGER.error("保存图片失败：%s（%s）", output_path, exc)
        return False


def _probe(
    camera_index: int,
    width: int,
    height: int,
    *,
    frames: int,
    tilt_threshold: float = DEFAULT_TILT_THRESHOLD_DEG,
    min_paper_ratio: float = DEFAULT_MIN_PAPER_AREA_RATIO,
    detect_pose: bool = True,
    as_json: bool = False,
) -> int:
    """无窗口抓取若干帧并输出分析摘要，便于在终端验证笔记本摄像头。"""
    started = time.monotonic()
    scanner = CameraScanner(
        CameraConfig(index=camera_index, width=width, height=height, warmup_frames=6, open_timeout=2.5)
    )
    if not scanner.open():
        message = scanner.last_error or f"无法打开摄像头 index={camera_index}"
        if as_json:
            print(json.dumps({"ok": False, "error": message}, ensure_ascii=False))
        else:
            print(f"PROBE_FAIL {message}")
        return 1

    status_counts = {status.value: 0 for status in ScanStatus}
    captured = 0
    angles: List[float] = []
    last_frame: Optional[np.ndarray] = None
    last_analysis: Optional[FrameAnalysis] = None
    last_status: Optional[ScanStatus] = None
    resolution = (0, 0)

    try:
        for index in range(max(1, int(frames))):
            frame = scanner.read_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            captured += 1
            last_frame = frame
            resolution = (frame.shape[1], frame.shape[0])
            analysis = scanner.analyze_frame(
                frame,
                tilt_threshold=tilt_threshold,
                min_paper_area_ratio=min_paper_ratio,
                detect_pose=detect_pose,
            )
            last_analysis = analysis
            status_counts[analysis.status.value] += 1
            if analysis.pose.found:
                angles.append(analysis.pose.angle_deg)

            if not as_json and (last_status is None or analysis.status is not last_status):
                tilt_text = f"{analysis.pose.angle_deg:+.1f}°" if analysis.pose.found else "n/a"
                LOGGER.info(
                    "frame %02d/%02d  status=%s  tilt=%s  qr=%s",
                    index + 1,
                    frames,
                    analysis.status.value,
                    tilt_text,
                    [code.data for code in analysis.codes],
                )
            last_status = analysis.status
            time.sleep(0.02)

        snapshot: Optional[Path] = None
        if last_frame is not None and last_analysis is not None:
            output_path = CAPTURE_DIR / f"probe_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            annotated = annotate_frame(
                last_frame,
                last_analysis.codes,
                show_hint=False,
                analysis=last_analysis,
            )
            if _imwrite_unicode(output_path, annotated):
                snapshot = output_path
    finally:
        scanner.release()

    elapsed = time.monotonic() - started
    result: Dict[str, Any] = {
        "ok": captured > 0,
        "camera": camera_index,
        "frames_requested": max(1, int(frames)),
        "frames_captured": captured,
        "resolution": list(resolution),
        "status_counts": status_counts,
        "tilt_range": [round(min(angles), 2), round(max(angles), 2)] if angles else [],
        "last_qr": last_analysis.code_data if last_analysis else None,
        "last_status": last_analysis.status.value if last_analysis else None,
        "needs_correction": last_analysis.needs_correction if last_analysis else False,
        "snapshot": str(snapshot) if snapshot is not None else None,
        "elapsed_sec": round(elapsed, 2),
    }

    if as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(
            "PROBE_OK "
            f"frames={captured}/{frames} "
            f"resolution={resolution[0]}x{resolution[1]} "
            f"last_status={result['last_status']} "
            f"tilt_range={result['tilt_range']} "
            f"last_qr={result['last_qr']} "
            f"elapsed={elapsed:.2f}s"
        )
        print(f"status_counts={status_counts}")
        if snapshot is not None:
            print(f"snapshot={snapshot}")
    return 0 if captured > 0 else 1
    parser = argparse.ArgumentParser(description="外星花园 · 摄像头与二维码识别工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号（默认 0）")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度（默认 1920）")
    parser.add_argument("--height", type=int, default=1080, help="采集高度（默认 1080）")
    parser.add_argument("--image", type=Path, default=None, help="离线识别指定图片")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="无摄像头演示模式：生成模拟画纸并实时识别二维码",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="探测 0~5 号摄像头并输出可用设备",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认 INFO）",
    )
    return parser.parse_args(argv)


def _build_test_paper(rotation_deg: float, qr_text: str = "ALIEN-GARDEN-DEMO-001") -> np.ndarray:
    """生成测试用“深色桌面 + 白纸 + 二维码”，可按指定角度旋转。"""
    width, height = 960, 720
    frame = np.full((height, width, 3), (24, 26, 28), dtype=np.uint8)

    paper_width, paper_height = 560, 396
    paper = np.full((paper_height, paper_width, 3), (242, 244, 238), dtype=np.uint8)
    cv2.rectangle(paper, (3, 3), (paper_width - 4, paper_height - 4), (198, 204, 196), 2)
    cv2.putText(
        paper,
        "MY ALIEN",
        (36, paper_height - 34),
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        1.2,
        (120, 70, 45),
        2,
        cv2.LINE_AA,
    )
    cv2.circle(paper, (150, 170), 46, (66, 158, 92), 4, cv2.LINE_AA)
    cv2.circle(paper, (134, 158), 6, (38, 45, 45), -1, cv2.LINE_AA)
    cv2.circle(paper, (166, 158), 6, (38, 45, 45), -1, cv2.LINE_AA)

    qr = _render_demo_qr(qr_text, 150)
    qr_height, qr_width = qr.shape[:2]
    paper[28 : 28 + qr_height, paper_width - qr_width - 28 : paper_width - 28] = qr

    # 先在深色底板上旋转，避免旋转裁掉纸角
    pad = 150
    canvas = np.full(
        (paper_height + 2 * pad, paper_width + 2 * pad, 3),
        (22, 22, 20),
        dtype=np.uint8,
    )
    canvas[pad : pad + paper_height, pad : pad + paper_width] = paper
    center = (canvas.shape[1] / 2.0, canvas.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        canvas,
        matrix,
        (canvas.shape[1], canvas.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(22, 22, 20),
    )

    x = (width - rotated.shape[1]) // 2
    y = (height - rotated.shape[0]) // 2
    frame[y : y + rotated.shape[0], x : x + rotated.shape[1]] = rotated
    return frame



def _selftest() -> int:
    """合成图算法自检：无需摄像头即可验证角度校准与联动状态。"""
    failures: List[str] = []
    scanner = CameraScanner()

    # rotation_deg 使用 OpenCV 约定：正值逆时针，因此检测角为负
    cases = (
        (0.0, 0.0, False),
        (-12.0, 12.0, True),
        (12.0, -12.0, True),
        (-3.0, 3.0, False),
    )
    for rotation, expected_angle, expected_correction in cases:
        frame = _build_test_paper(rotation)
        pose = detect_paper_pose(frame, tilt_threshold=8.0)
        angle_error = abs(pose.angle_deg - expected_angle) if pose.found else float("inf")
        passed = (
            pose.found
            and angle_error <= 3.0
            and pose.needs_correction is expected_correction
        )
        print(
            f"[POSE] rotation={rotation:+.1f} found={pose.found} "
            f"angle={pose.angle_deg:+.1f} expected={expected_angle:+.1f} "
            f"correction={pose.needs_correction} -> {'OK' if passed else 'FAIL'}"
        )
        if not passed:
            failures.append(f"pose rotation={rotation:+.1f}")

    blank = np.full((480, 640, 3), 18, dtype=np.uint8)
    blank_pose = detect_paper_pose(blank, tilt_threshold=8.0)
    if blank_pose.found:
        failures.append("blank frame should not contain paper")
    print(f"[POSE] blank found={blank_pose.found} -> {'OK' if not blank_pose.found else 'FAIL'}")

    if PYZBAR_AVAILABLE:
        level_frame = _build_test_paper(0.0)
        level_analysis = scanner.analyze_frame(level_frame, tilt_threshold=8.0)
        level_ok = level_analysis.ready and level_analysis.code_data == "ALIEN-GARDEN-DEMO-001"
        print(
            f"[LINK] level status={level_analysis.status.value} "
            f"qr={level_analysis.code_data} -> {'OK' if level_ok else 'FAIL'}"
        )
        if not level_ok:
            failures.append("level paper should be ready with QR")

        tilted_analysis = scanner.analyze_frame(_build_test_paper(15.0), tilt_threshold=8.0)
        tilted_ok = tilted_analysis.status is ScanStatus.TILTED and tilted_analysis.needs_correction
        print(
            f"[LINK] tilted status={tilted_analysis.status.value} "
            f"qr={tilted_analysis.code_data} -> {'OK' if tilted_ok else 'FAIL'}"
        )
        if not tilted_ok:
            failures.append("tilted paper must require correction")
    else:
        print(f"[LINK] pyzbar unavailable, skip QR checks: {PYZBAR_IMPORT_ERROR}")

    if failures:
        print(f"SELFTEST_FAIL {failures}")
        return 1
    print("SELFTEST_OK")
    return 0



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外星花园 · 摄像头识别与图纸姿态校准工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号（默认 0）")
    parser.add_argument("--width", type=int, default=1280, help="采集宽度（默认 1280）")
    parser.add_argument("--height", type=int, default=720, help="采集高度（默认 720）")
    parser.add_argument("--image", type=Path, default=None, help="离线识别指定图片")
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
        "--no-pose",
        action="store_true",
        help="关闭图纸轮廓校准，仅做二维码识别",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="无窗口抓取摄像头画面，在终端输出识别与倾斜角摘要",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=30,
        help="--probe 抓取帧数（默认 30）",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="--probe 以 JSON 输出结果",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="合成图算法自检（不需要摄像头）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="无摄像头演示模式：生成模拟画纸并实时识别二维码",
    )
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="探测 0~5 号摄像头并输出可用设备",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="日志级别（默认 INFO）",
    )
    return parser.parse_args(argv)



def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    detect_pose = not args.no_pose

    if args.selftest:
        return _selftest()

    if not PYZBAR_AVAILABLE:
        LOGGER.warning("pyzbar 初始化失败：%s", PYZBAR_IMPORT_ERROR)
        LOGGER.warning("摄像头预览仍可用，但二维码识别会返回空结果。")

    if args.demo:
        return _demo(
            min(args.width, 1280),
            min(args.height, 720),
            tilt_threshold=args.tilt_threshold,
            min_paper_ratio=args.min_paper_ratio,
            detect_pose=detect_pose,
        )

    if args.probe:
        return _probe(
            args.camera,
            args.width,
            args.height,
            frames=args.frames,
            tilt_threshold=args.tilt_threshold,
            min_paper_ratio=args.min_paper_ratio,
            detect_pose=detect_pose,
            as_json=args.json,
        )

    if args.list_cameras:
        cameras = find_available_cameras()
        if not cameras:
            print("未发现可用摄像头。")
            return 1
        for index, width, height in cameras:
            print(f"camera index={index}  {width}x{height}")
        return 0

    if args.image is not None:
        if not args.image.is_file():
            print(f"图片不存在：{args.image}")
            return 2
        return _decode_image(
            args.image,
            tilt_threshold=args.tilt_threshold,
            min_paper_ratio=args.min_paper_ratio,
            detect_pose=detect_pose,
        )

    return _preview(
        args.camera,
        args.width,
        args.height,
        tilt_threshold=args.tilt_threshold,
        min_paper_ratio=args.min_paper_ratio,
        detect_pose=detect_pose,
    )


if __name__ == "__main__":
    raise SystemExit(main())


# __END__
