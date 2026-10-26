"""视觉链路联调脚本（摄像头实时帧 + pyzbar 二维码 + 图纸水平角）。

与 ``tests/test_vision.py`` 中的单元测试不同，本脚本用于在真实设备上做一次
端到端联调，并额外用合成图纸做确定性自检，即使现场没有图纸也能得到明确结论：

1. 打开摄像头抓取实时帧（无摄像头时可用 ``--demo`` 或自动回退到合成画面）；
2. 用 OpenCV 灰度化 + pyzbar 解码二维码，并与 ``scanner.CameraScanner`` 交叉校验；
3. 用轮廓检测估算图纸摆放角，超过阈值时打印“请顺时针 / 逆时针旋转图纸”提示；
4. 脚本位于项目根目录，会自动把根目录加入 ``sys.path``，可直接运行。

用法::

    python test_vision.py                          # 摄像头实时预览，q / ESC 退出
    python test_vision.py --frames 60 --no-window  # 无窗口抓 60 帧后自动结束
    python test_vision.py --demo                   # 无摄像头：合成画纸 + 二维码
    python test_vision.py --help

退出码：0 表示本次联调通过，1 表示存在必需项失败。
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

# 本脚本必须能独立运行：先确保项目根目录在 sys.path 中，再导入项目模块。
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import scanner  # noqa: E402  （必须在上面的 sys.path 设置之后导入）

LOGGER = logging.getLogger("alien_garden.test_vision")
QR_TEXT = "ALIEN-GARDEN-DEMO-001"
PREVIEW_WINDOW = "Alien Garden - test_vision (q/ESC, s: snapshot)"
DEFAULT_HEADLESS_FRAMES = 90
PYZBAR_CHECK_EVERY = 5

# pyzbar 可能因缺少 libzbar DLL 而导入失败，此时脚本仍会给出明确失败项。
try:
    from pyzbar.pyzbar import ZBarSymbol, decode as zbar_decode

    PYZBAR_AVAILABLE = True
    PYZBAR_IMPORT_ERROR = ""
except Exception as exc:  # ImportError 或 OSError（DLL 缺失）
    zbar_decode = None  # type: ignore[assignment]
    ZBarSymbol = None  # type: ignore[assignment]
    PYZBAR_AVAILABLE = False
    PYZBAR_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


@dataclass
class Check:
    """一条联调检查结果。"""

    name: str
    ok: bool
    detail: str
    required: bool = True


def record(
    checks: List[Check],
    name: str,
    ok: bool,
    detail: str,
    *,
    required: bool = True,
) -> None:
    """记录并即时打印一条检查结果；非必需项只作为提示。"""
    checks.append(Check(name=name, ok=bool(ok), detail=detail, required=required))
    if not required:
        tag = "INFO"
    else:
        tag = "PASS" if ok else "FAIL"
    print(f"[CHECK] {tag} {name}：{detail}")


def decode_qr_direct(frame: np.ndarray) -> List[str]:
    """OpenCV 灰度化 + pyzbar 直接解码，作为 scanner 结果的独立对照。"""
    if not PYZBAR_AVAILABLE or zbar_decode is None or ZBarSymbol is None:
        return []
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    try:
        items = zbar_decode(gray, symbols=[ZBarSymbol.QRCODE])
    except Exception as exc:  # 解码失败不能中断联调
        LOGGER.warning("pyzbar 直接解码失败：%s: %s", type(exc).__name__, exc)
        return []
    return [item.data.decode("utf-8", errors="replace") for item in items]


def run_synthetic_checks(
    tilt_threshold: float,
    min_paper_ratio: float,
) -> List[Check]:
    """用合成图纸验证二维码识别与轮廓倾斜角，结果与现场环境无关。"""
    print("\n=== 1/2 合成图纸算法自检 ===")
    checks: List[Check] = []
    analyzer = scanner.CameraScanner()

    if not PYZBAR_AVAILABLE:
        record(checks, "pyzbar 可用", False, f"导入失败：{PYZBAR_IMPORT_ERROR}")
        return checks

    # 水平图纸：二维码可识别、轮廓角度接近 0、状态 READY
    level_frame = scanner._build_test_paper(0.0, qr_text=QR_TEXT)
    direct_codes = decode_qr_direct(level_frame)
    record(
        checks,
        "pyzbar 直接解码",
        QR_TEXT in direct_codes,
        f"解码结果={direct_codes}",
    )

    level_pose = scanner.detect_paper_pose(level_frame, tilt_threshold=tilt_threshold)
    level_angle_ok = level_pose.found and abs(level_pose.angle_deg) <= 2.0
    record(
        checks,
        "水平图纸轮廓角度",
        level_angle_ok and not level_pose.needs_correction,
        f"found={level_pose.found} angle={level_pose.angle_deg:+.1f}° "
        f"area={level_pose.area_ratio:.1%} needs_correction={level_pose.needs_correction}",
    )

    level_analysis = analyzer.analyze_frame(
        level_frame,
        tilt_threshold=tilt_threshold,
        min_paper_area_ratio=min_paper_ratio,
    )
    record(
        checks,
        "水平图纸联动状态",
        level_analysis.status is scanner.ScanStatus.READY
        and level_analysis.code_data == QR_TEXT,
        f"status={level_analysis.status.value} qr={level_analysis.code_data}",
    )

    # 倾斜 15° 的图纸：必须判定 TILTED，并给出回正方向提示
    tilted_frame = scanner._build_test_paper(15.0, qr_text=QR_TEXT)
    tilted_pose = scanner.detect_paper_pose(tilted_frame, tilt_threshold=tilt_threshold)
    tilted_analysis = analyzer.analyze_frame(
        tilted_frame,
        tilt_threshold=tilt_threshold,
        min_paper_area_ratio=min_paper_ratio,
    )
    hint = tilted_pose.correction_hint or tilted_analysis.message
    tilt_ok = (
        tilted_pose.found
        and tilted_pose.needs_correction
        and tilted_analysis.status is scanner.ScanStatus.TILTED
        and "旋转" in hint
    )
    record(
        checks,
        "倾斜图纸校正提示",
        tilt_ok,
        f"angle={tilted_pose.angle_deg:+.1f}° "
        f"status={tilted_analysis.status.value} 提示={hint}",
    )
    if hint:
        print(f"[HINT] {hint}")

    # 纯色画面不应误检图纸
    blank_frame = np.full((480, 640, 3), 20, dtype=np.uint8)
    blank_pose = scanner.detect_paper_pose(blank_frame, tilt_threshold=tilt_threshold)
    record(checks, "空画面不误检图纸", not blank_pose.found, f"found={blank_pose.found}")
    return checks


def run_camera_checks(
    *,
    camera_index: int,
    frames: int,
    demo: bool,
    preview: bool,
    tilt_threshold: float,
    min_paper_ratio: float,
) -> List[Check]:
    """对实时摄像头（或 ``--demo`` 合成画面）逐帧联调。

    ``demo=True`` 时所有识别项都是必需项；接真实摄像头时，若观众尚未把图纸
    放入画面，识别为空属于正常现象，因此这些项降级为提示项，只把「摄像头能
    正常出帧、算法能正常执行」作为必需项。
    """

    print("\n=== 2/2 实时画面联调 ===")
    checks: List[Check] = []
    total = max(0, int(frames))
    analyzer = scanner.CameraScanner(scanner.CameraConfig(index=camera_index))
    detection_required = demo

    scanner_codes: Counter[str] = Counter()
    direct_codes: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    angles: List[float] = []
    first_ready: Optional[scanner.FrameAnalysis] = None
    last_analysis: Optional[scanner.FrameAnalysis] = None
    analyzed = 0
    mismatches = 0
    started = time.monotonic()

    if demo:
        record(checks, "演示画面生成", True, "使用合成画纸 + 二维码，无需真实摄像头")
    elif not analyzer.open():
        record(
            checks,
            "摄像头打开",
            False,
            f"index={camera_index}：{analyzer.last_error or '打开失败'}",
            required=False,
        )
        print("[HINT] 没有可用摄像头时，可执行 python test_vision.py --demo 验证算法链路。")
        return checks
    else:
        record(checks, "摄像头打开", True, f"index={camera_index} 已连接")

    try:
        index = 0
        while total == 0 or index < total:
            if demo:
                frame = scanner.build_demo_frame(elapsed=time.monotonic() - started)
            else:
                frame = analyzer.read_frame()
                if frame is None:
                    record(
                        checks,
                        "摄像头读取帧",
                        False,
                        analyzer.last_error or "摄像头读取失败",
                    )
                    break

            try:
                analysis = analyzer.analyze_frame(
                    frame,
                    tilt_threshold=tilt_threshold,
                    min_paper_area_ratio=min_paper_ratio,
                )
            except Exception as exc:  # 单帧异常不能中断联调
                record(checks, "帧分析", False, f"{type(exc).__name__}: {exc}")
                break

            analyzed += 1
            status_counter[analysis.status.value] += 1
            for code in analysis.codes:
                scanner_codes[code.data] += 1
            if analysis.pose.found:
                angles.append(analysis.pose.angle_deg)
            if analysis.ready and first_ready is None:
                first_ready = analysis
            last_analysis = analysis

            # 每 N 帧用 pyzbar 直接解码做一次交叉校验
            if PYZBAR_AVAILABLE and index % PYZBAR_CHECK_EVERY == 0:
                direct = decode_qr_direct(frame)
                direct_codes.update(direct)
                if set(direct) != {code.data for code in analysis.codes}:
                    mismatches += 1

            if preview:
                annotated = scanner.annotate_frame(
                    frame,
                    analysis.codes,
                    analysis=analysis,
                )
                cv2.imshow(PREVIEW_WINDOW, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    print("[INFO] 用户退出预览")
                    break
                if key == ord("s"):
                    snapshot = scanner.CAPTURE_DIR / f"test_vision_{int(time.time())}.jpg"
                    if scanner._imwrite_unicode(snapshot, annotated):
                        print(f"[INFO] 快照已保存：{snapshot}")

            index += 1
    except KeyboardInterrupt:
        print("\n[INFO] 用户中断实时联调")
    finally:
        if preview:
            cv2.destroyAllWindows()
        analyzer.release()

    record(
        checks,
        "摄像头逐帧分析",
        analyzed > 0,
        f"分析 {analyzed} 帧" + (f" / 计划 {total} 帧" if total else "（实时预览）"),
    )

    qr_detail = (
        f"{sum(scanner_codes.values())} 次 / {len(scanner_codes)} 种：{dict(scanner_codes)}"
        if scanner_codes
        else "画面中暂未识别到二维码"
    )
    record(checks, "实时二维码识别", bool(scanner_codes), qr_detail, required=detection_required)

    if angles:
        angle_detail = (
            f"{len(angles)} 帧检出轮廓，角度范围 "
            f"{min(angles):+.1f}° ~ {max(angles):+.1f}°（阈值 ±{tilt_threshold:.1f}°）"
        )
    else:
        angle_detail = "画面中暂未检测到图纸轮廓"
    record(checks, "实时图纸轮廓", bool(angles), angle_detail, required=detection_required)

    if first_ready is not None:
        ready_detail = f"status={first_ready.status.value} qr={first_ready.code_data}"
    else:
        ready_detail = f"尚未出现 READY，状态分布：{dict(status_counter)}"
    record(checks, "实时联动状态", first_ready is not None, ready_detail, required=detection_required)

    if PYZBAR_AVAILABLE:
        record(
            checks,
            "pyzbar 交叉校验",
            mismatches == 0 and bool(direct_codes),
            f"直接解码 {dict(direct_codes)}，与 scanner 不一致帧数 {mismatches}",
            required=False,
        )
    else:
        record(checks, "pyzbar 交叉校验", False, f"pyzbar 不可用：{PYZBAR_IMPORT_ERROR}", required=False)

    if last_analysis is not None and last_analysis.pose.needs_correction:
        hint = last_analysis.pose.correction_hint
        print(f"[HINT] {hint}")
        record(
            checks,
            "倾斜校正提示",
            "旋转" in hint,
            f"angle={last_analysis.pose.angle_deg:+.1f}° 提示={hint}",
            required=False,
        )

    print(f"[INFO] 状态分布：{dict(status_counter)}")
    return checks


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="外星花园视觉链路联调：摄像头 + pyzbar 二维码 + 图纸水平角",
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头序号（默认 0）")
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help=(
            "联调帧数；0 表示自动（带窗口时持续预览到按键退出，"
            f"--no-window 时抓取 {DEFAULT_HEADLESS_FRAMES} 帧）"
        ),
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="不弹出预览窗口，抓取指定帧数后自动结束",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="无摄像头：使用合成画纸 + 二维码",
    )
    parser.add_argument(
        "--tilt-threshold",
        type=float,
        default=scanner.DEFAULT_TILT_THRESHOLD_DEG,
        help=f"图纸允许倾斜角，单位度（默认 {scanner.DEFAULT_TILT_THRESHOLD_DEG:.0f}）",
    )
    parser.add_argument(
        "--min-paper-ratio",
        type=float,
        default=scanner.DEFAULT_MIN_PAPER_AREA_RATIO,
        help=f"图纸最小面积占比（默认 {scanner.DEFAULT_MIN_PAPER_AREA_RATIO:.2f}）",
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

    print("外星花园视觉链路联调：合成自检 + 实时画面")
    checks = run_synthetic_checks(args.tilt_threshold, args.min_paper_ratio)
    frames = args.frames
    if frames <= 0:
        frames = 0 if not args.no_window else DEFAULT_HEADLESS_FRAMES
    checks.extend(
        run_camera_checks(
            camera_index=args.camera,
            frames=frames,
            demo=args.demo,
            preview=not args.no_window,
            tilt_threshold=args.tilt_threshold,
            min_paper_ratio=args.min_paper_ratio,
        )
    )

    required = [check for check in checks if check.required]
    failed = [check for check in required if not check.ok]

    print("\n=== 联调汇总 ===")
    for check in checks:
        if not check.required:
            tag = "INFO"
        else:
            tag = "PASS" if check.ok else "FAIL"
        print(f"[{tag}] {check.name}：{check.detail}")
    print(f"共 {len(checks)} 项，必需项 {len(required)} 项，失败 {len(failed)} 项")

    if failed:
        print("结论：视觉链路存在必需项失败，请按上面的 FAIL 项排查。")
        return 1
    print("结论：视觉链路联调通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

