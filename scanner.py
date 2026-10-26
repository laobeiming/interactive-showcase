"""摄像头采集与二维码识别模块（阶段一骨架）。

本模块只负责“看得见、读得出”：
* 封装 OpenCV 摄像头的打开、读取、释放与热插拔容错；
* 封装 pyzbar 二维码识别，输出二维码内容、位置与中心点；
* 提供独立命令行预览工具，便于现场调试高拍仪。

后续阶段可直接复用 ``CameraScanner.scan_frame()``，把识别到的二维码 ID
交给纸张姿态检测与场景联动模块。

命令行示例：
    python scanner.py                     # 默认摄像头预览
    python scanner.py --camera 1          # 指定第二路摄像头
    python scanner.py --image qr.png      # 离线识别单张图片
    python scanner.py --list-cameras      # 探测可用摄像头

预览窗口按键：q / ESC 退出，s 保存当前帧快照。
"""

from __future__ import annotations

import argparse
import logging
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("alien_garden.scanner")
IS_WINDOWS = platform.system() == "Windows"
PROJECT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = PROJECT_DIR / "captures"

Point = Tuple[int, int]
Rect = Tuple[int, int, int, int]

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


@dataclass
class CameraConfig:
    """摄像头采集参数，后续可由配置文件覆盖。"""

    index: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    warmup_frames: int = 5


class CameraScanner:
    """摄像头 + 二维码识别的统一入口。

    典型用法::

        with CameraScanner(CameraConfig(index=0)) as scanner:
            frame, codes = scanner.scan_once()
            for code in codes:
                print(code.data, code.center)

    该类不依赖 Qt，可在主线程之外复用；但 OpenCV 的窗口操作需留在主线程。
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        self.config = config or CameraConfig()
        self._capture: Optional[cv2.VideoCapture] = None
        self.last_error: Optional[str] = None

    # ---- 设备管理 -----------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._capture is not None and self._capture.isOpened()

    def open(self) -> bool:
        """打开摄像头并完成预热，成功返回 True。"""
        if self.is_open:
            return True

        self.last_error = None
        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        LOGGER.debug("尝试打开摄像头 index=%s backend=%s", self.config.index, backend)
        capture = cv2.VideoCapture(self.config.index, backend)

        # DirectShow 偶发失败时，退回系统默认后端再试一次
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            LOGGER.debug("DirectShow 打开失败，尝试系统默认后端")
            capture = cv2.VideoCapture(self.config.index)

        if not capture.isOpened():
            capture.release()
            self.last_error = f"无法打开摄像头 index={self.config.index}（设备被占用或未连接）"
            LOGGER.error(self.last_error)
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        capture.set(cv2.CAP_PROP_FPS, self.config.fps)
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 尽量取最新帧，降低识别延迟

        self._capture = capture
        for _ in range(max(0, self.config.warmup_frames)):
            capture.read()

        actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        LOGGER.info(
            "摄像头已打开：index=%s，分辨率=%sx%s，FPS=%.1f",
            self.config.index,
            actual_width,
            actual_height,
            actual_fps,
        )
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """读取一帧 BGR 图像；失败时返回 None 并更新 last_error。"""
        if not self.is_open and not self.open():
            return None

        assert self._capture is not None
        ok, frame = self._capture.read()
        if not ok or frame is None:
            self.last_error = "摄像头读取失败（设备被占用或已断开）"
            LOGGER.warning(self.last_error)
            return None
        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            LOGGER.info("摄像头已释放")

    # ---- 二维码识别 ---------------------------------------------------
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


def _ascii_label(text: str, limit: int = 48) -> str:
    """OpenCV 的 putText 只支持 ASCII，预览窗口里把非 ASCII 字符替换掉。"""
    safe = text.encode("ascii", errors="replace").decode("ascii")
    return safe if len(safe) <= limit else safe[: limit - 3] + "..."


def annotate_frame(
    frame: np.ndarray,
    codes: Sequence[QRCode],
    *,
    show_hint: bool = True,
) -> np.ndarray:
    """在图像副本上框出二维码，返回用于预览的 BGR 图像。"""
    canvas = frame.copy()
    for code in codes:
        x, y, width, height = code.rect
        if len(code.polygon) >= 4:
            points = np.array(code.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [points], True, (0, 255, 170), 3, cv2.LINE_AA)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (0, 220, 150), 2)
        cv2.circle(canvas, code.center, 5, (0, 255, 200), -1, cv2.LINE_AA)

        label = f"QR: {_ascii_label(code.data)}"
        label_y = max(24, y - 12)
        cv2.putText(
            canvas,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 170),
            2,
            cv2.LINE_AA,
        )

    if show_hint:
        cv2.putText(
            canvas,
            "q / ESC: quit    s: snapshot",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "q / ESC: quit    s: snapshot",
            (16, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (235, 245, 240),
            1,
            cv2.LINE_AA,
        )
    return canvas


def find_available_cameras(max_index: int = 5) -> List[Tuple[int, int, int]]:
    """探测前 ``max_index`` 路摄像头，返回 ``[(index, width, height), ...]``。"""
    available: List[Tuple[int, int, int]] = []
    for index in range(max_index + 1):
        backend = cv2.CAP_DSHOW if IS_WINDOWS else cv2.CAP_ANY
        capture = cv2.VideoCapture(index, backend)
        if not capture.isOpened() and backend != cv2.CAP_ANY:
            capture.release()
            capture = cv2.VideoCapture(index)
        try:
            if not capture.isOpened():
                continue
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            height, width = frame.shape[:2]
            available.append((index, width, height))
        finally:
            capture.release()
    return available


def _decode_image(image_path: Path) -> int:
    """离线识别单张图片，并把标注结果保存到 captures/。"""
    if not PYZBAR_AVAILABLE:
        LOGGER.error("pyzbar 不可用：%s", PYZBAR_IMPORT_ERROR)
        return 2

    frame = cv2.imread(str(image_path))
    if frame is None:
        LOGGER.error("无法读取图片：%s", image_path)
        return 2

    scanner = CameraScanner()
    codes = scanner.scan_frame(frame)
    annotated = annotate_frame(frame, codes, show_hint=False)

    CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = CAPTURE_DIR / f"annotated_{image_path.stem}_{int(time.time())}.jpg"
    cv2.imwrite(str(output_path), annotated)

    for code in codes:
        print(f"[QR] {code.data}  rect={code.rect}  center={code.center}")
    print(f"识别到 {len(codes)} 个二维码，标注结果已保存：{output_path}")
    if not codes and scanner.last_error:
        print(f"提示：{scanner.last_error}")
    return 0 if codes else 1


def _preview(camera_index: int, width: int, height: int) -> int:
    """打开摄像头实时预览并识别二维码。"""
    scanner = CameraScanner(CameraConfig(index=camera_index, width=width, height=height))
    if not scanner.open():
        print(f"无法打开摄像头 {camera_index}。请检查连接或改用 --list-cameras 探测。")
        return 1
    if not PYZBAR_AVAILABLE:
        LOGGER.warning("pyzbar 不可用（%s），仅显示摄像头画面", PYZBAR_IMPORT_ERROR)

    window_name = "Alien Garden - Scanner Preview"
    last_seen: dict[str, float] = {}
    try:
        while True:
            frame = scanner.read_frame()
            if frame is None:
                break

            codes = scanner.scan_frame(frame)
            now = time.monotonic()
            for code in codes:
                if now - last_seen.get(code.data, 0.0) > 2.0:
                    last_seen[code.data] = now
                    LOGGER.info("识别到二维码：%s（中心点 %s）", code.data, code.center)

            cv2.imshow(window_name, annotate_frame(frame, codes))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
                output_path = CAPTURE_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(output_path), frame)
                LOGGER.info("快照已保存：%s", output_path)
    except KeyboardInterrupt:
        LOGGER.info("用户中断预览")
    finally:
        scanner.release()
        cv2.destroyAllWindows()
    return 0


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="外星花园 · 摄像头与二维码识别工具")
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号（默认 0）")
    parser.add_argument("--width", type=int, default=1920, help="采集宽度（默认 1920）")
    parser.add_argument("--height", type=int, default=1080, help="采集高度（默认 1080）")
    parser.add_argument("--image", type=Path, default=None, help="离线识别指定图片")
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

    if not PYZBAR_AVAILABLE:
        LOGGER.warning("pyzbar 初始化失败：%s", PYZBAR_IMPORT_ERROR)
        LOGGER.warning("摄像头预览仍可用，但二维码识别会返回空结果。")

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
        return _decode_image(args.image)

    return _preview(args.camera, args.width, args.height)


if __name__ == "__main__":
    raise SystemExit(main())


# __END__
