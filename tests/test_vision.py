"""阶段二视觉检测与联动逻辑的自动化测试。

运行：
    python -m unittest discover -s tests -v

测试覆盖：
* 轮廓检测估算图纸倾斜角（水平 / 顺时针 / 逆时针）；
* 二维码 + 图纸水平 -> READY，倾斜 -> TILTED（不触发联动）；
* Windows 中文路径图片读写；
* main.py 的场景联动触发条件与稳定帧数；
* 二维码 ID 到联动主题的映射。
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

# 单元测试不弹出真实窗口
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault(
    "QT_QPA_FONTDIR",
    os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts"),
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import main as showcase_main  # noqa: E402
import scanner  # noqa: E402


class PaperPoseTests(unittest.TestCase):
    """图纸水平角校准算法测试。"""

    def test_level_paper(self) -> None:
        frame = scanner._build_test_paper(0.0)
        pose = scanner.detect_paper_pose(frame, tilt_threshold=8.0)
        self.assertTrue(pose.found)
        self.assertAlmostEqual(pose.angle_deg, 0.0, delta=2.0)
        self.assertFalse(pose.needs_correction)
        self.assertEqual(pose.correction_hint, "")

    def test_clockwise_paper_needs_correction(self) -> None:
        # OpenCV 旋转 -12° 表示屏幕上的顺时针倾斜
        frame = scanner._build_test_paper(-12.0)
        pose = scanner.detect_paper_pose(frame, tilt_threshold=8.0)
        self.assertTrue(pose.found)
        self.assertAlmostEqual(pose.angle_deg, 12.0, delta=2.5)
        self.assertTrue(pose.needs_correction)
        self.assertIn("逆时针", pose.correction_hint)

    def test_counter_clockwise_paper_needs_correction(self) -> None:
        frame = scanner._build_test_paper(14.0)
        pose = scanner.detect_paper_pose(frame, tilt_threshold=8.0)
        self.assertTrue(pose.found)
        self.assertAlmostEqual(pose.angle_deg, -14.0, delta=2.5)
        self.assertTrue(pose.needs_correction)
        self.assertIn("顺时针", pose.correction_hint)

    def test_blank_frame_has_no_paper(self) -> None:
        import numpy as np

        frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        pose = scanner.detect_paper_pose(frame, tilt_threshold=8.0)
        self.assertFalse(pose.found)
        self.assertFalse(pose.needs_correction)


class ScanStatusTests(unittest.TestCase):
    """二维码 + 图纸姿态的综合状态测试。"""

    def setUp(self) -> None:
        self.camera = scanner.CameraScanner()

    def test_level_paper_with_qr_is_ready(self) -> None:
        frame = scanner._build_test_paper(0.0)
        analysis = self.camera.analyze_frame(frame, tilt_threshold=8.0)
        self.assertEqual(analysis.status, scanner.ScanStatus.READY)
        self.assertEqual(analysis.code_data, "ALIEN-GARDEN-DEMO-001")
        self.assertFalse(analysis.needs_correction)
        self.assertTrue(analysis.ready)

    def test_tilted_paper_with_qr_is_not_ready(self) -> None:
        frame = scanner._build_test_paper(15.0)
        analysis = self.camera.analyze_frame(frame, tilt_threshold=8.0)
        self.assertEqual(analysis.status, scanner.ScanStatus.TILTED)
        self.assertTrue(analysis.needs_correction)
        self.assertFalse(analysis.ready)
        self.assertIn("请", analysis.message)

    def test_tilt_threshold_is_configurable(self) -> None:
        frame = scanner._build_test_paper(6.0)
        strict = self.camera.analyze_frame(frame, tilt_threshold=3.0)
        loose = self.camera.analyze_frame(frame, tilt_threshold=10.0)
        self.assertEqual(strict.status, scanner.ScanStatus.TILTED)
        self.assertEqual(loose.status, scanner.ScanStatus.READY)


class UnicodeImageTests(unittest.TestCase):
    """Windows 中文路径下的图片读写回归测试。"""

    def test_unicode_path_roundtrip(self) -> None:
        frame = scanner._build_test_paper(0.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "中文目录" / "测试图纸.jpg"
            self.assertTrue(scanner._imwrite_unicode(output_path, frame))
            self.assertTrue(output_path.is_file())

            loaded = scanner._imread_unicode(output_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.shape, frame.shape)


class SceneThemeTests(unittest.TestCase):
    """二维码 ID -> 场景主题映射测试。"""

    def test_known_theme(self) -> None:
        theme = showcase_main.resolve_scene_theme("ALIEN-GARDEN-DEMO-001")
        self.assertEqual(theme.key, "garden")
        self.assertEqual(theme.title, "外星花园")

    def test_ocean_theme(self) -> None:
        theme = showcase_main.resolve_scene_theme("OCEAN-2024")
        self.assertEqual(theme.key, "ocean")

    def test_unknown_theme_is_stable(self) -> None:
        first = showcase_main.resolve_scene_theme("CUSTOM-XYZ-99")
        second = showcase_main.resolve_scene_theme("CUSTOM-XYZ-99")
        self.assertEqual(first.key, "custom")
        self.assertEqual(first.title, "神秘星球")
        self.assertEqual(first.accent, second.accent)



class LinkageTests(unittest.TestCase):
    """main.py 联动触发条件测试（不打开真实摄像头）。"""

    @classmethod
    def setUpClass(cls) -> None:
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        cls.camera = scanner.CameraScanner()
        cls.level_frame = scanner._build_test_paper(0.0)
        cls.level_analysis = cls.camera.analyze_frame(cls.level_frame, tilt_threshold=8.0)
        cls.tilted_frame = scanner._build_test_paper(15.0)
        cls.tilted_analysis = cls.camera.analyze_frame(cls.tilted_frame, tilt_threshold=8.0)

    def setUp(self) -> None:
        self.window = showcase_main.StandbyWindow(
            video_path=None,
            windowed=True,
            force_fallback=True,
            camera_enabled=False,
            scene_seconds=30.0,
        )
        self.window._stack.setCurrentWidget(self.window._camera_canvas)

    def tearDown(self) -> None:
        self.window.close()

    def test_ready_state_triggers_after_stable_frames(self) -> None:
        self.assertEqual(self.level_analysis.status, scanner.ScanStatus.READY)

        for _ in range(showcase_main.TRIGGER_STABLE_FRAMES - 1):
            self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.assertEqual(self.window._trigger_count, 0)

        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.assertEqual(self.window._trigger_count, 1)
        self.assertIs(self.window._stack.currentWidget(), self.window._scene_canvas)
        self.assertIsNotNone(self.window._scene_canvas.theme)
        assert self.window._scene_canvas.theme is not None
        self.assertEqual(self.window._scene_canvas.theme.key, "garden")

    def test_tilted_state_never_triggers(self) -> None:
        self.assertEqual(self.tilted_analysis.status, scanner.ScanStatus.TILTED)

        for _ in range(showcase_main.TRIGGER_STABLE_FRAMES * 3):
            self.window._on_frame_ready(self.tilted_frame, self.tilted_analysis)
        self.assertEqual(self.window._trigger_count, 0)
        self.assertIs(self.window._stack.currentWidget(), self.window._camera_canvas)

    def test_tilted_frame_resets_ready_streak(self) -> None:
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.window._on_frame_ready(self.tilted_frame, self.tilted_analysis)
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.assertEqual(self.window._trigger_count, 0)

        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self.assertEqual(self.window._trigger_count, 1)

    def test_camera_and_scene_canvases_render(self) -> None:
        self.window.resize(1280, 720)
        self.window._camera_canvas.resize(1280, 720)
        self.window._camera_canvas.set_frame(self.level_frame, self.level_analysis)
        camera_image = self.window._camera_canvas.grab()
        self.assertFalse(camera_image.isNull())
        self.assertGreater(camera_image.width(), 0)

        self.window._scene_canvas.resize(1280, 720)
        theme = showcase_main.resolve_scene_theme("ALIEN-GARDEN-DEMO-001")
        self.window._scene_canvas.set_scene(
            theme,
            "ALIEN-GARDEN-DEMO-001",
            showcase_main.bgr_to_qimage(self.level_frame),
            10.0,
        )
        scene_image = self.window._scene_canvas.grab()
        self.assertFalse(scene_image.isNull())
        self.assertGreater(scene_image.width(), 0)



class EntryPointTests(unittest.TestCase):
    """main.py 统一入口的默认参数测试。"""

    def test_camera_and_fullscreen_are_default(self) -> None:
        args = showcase_main.parse_args([])
        self.assertFalse(args.no_camera)
        self.assertFalse(args.windowed)

    def test_no_camera_flag_disables_scanner(self) -> None:
        args = showcase_main.parse_args(["--no-camera", "--windowed"])
        self.assertTrue(args.no_camera)
        self.assertTrue(args.windowed)

    def test_start_camera_flag_remains_compatible(self) -> None:
        args = showcase_main.parse_args(["--start-camera"])
        self.assertTrue(args.start_camera)
        self.assertFalse(args.no_camera)

