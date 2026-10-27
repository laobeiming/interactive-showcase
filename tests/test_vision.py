"""阶段二视觉检测与联动逻辑的自动化测试。

运行：
    python -m unittest discover -s tests -v

测试覆盖：
* 轮廓检测估算图纸倾斜角（水平 / 顺时针 / 逆时针）；
* 二维码 + 图纸水平 -> READY，倾斜 -> TILTED（不触发联动）；
* main.py 的 Space / Enter 按键校验、拍照保存与 8 秒复位状态机；
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


class AlignmentPayloadTests(unittest.TestCase):
    """旋转外接矩形角度标准化与对外精简结果测试。"""

    def test_alignment_payload_and_tuple_unpack(self) -> None:
        frame = scanner._build_test_paper(-12.0)
        analysis = scanner.CameraScanner().analyze_frame(frame, tilt_threshold=5.0)

        payload = analysis.to_alignment()
        self.assertEqual(set(payload), {"is_straight", "angle", "code_id"})
        self.assertFalse(payload["is_straight"])
        self.assertAlmostEqual(payload["angle"], 12.0, delta=2.0)
        self.assertEqual(payload["code_id"], "ALIEN-GARDEN-DEMO-001")

        is_straight, angle, code_id = analysis
        self.assertFalse(is_straight)
        self.assertAlmostEqual(angle, 12.0, delta=2.0)
        self.assertEqual(code_id, "ALIEN-GARDEN-DEMO-001")

        self.assertFalse(analysis.is_straight)
        self.assertTrue(analysis.pose.needs_correction)
        self.assertIn("请摆正图画", analysis.pose.status_label)

    def test_min_area_rect_metadata(self) -> None:
        frame = scanner._build_test_paper(0.0)
        pose = scanner.detect_paper_pose(frame, tilt_threshold=5.0)

        self.assertTrue(pose.found)
        self.assertIsNotNone(pose.rect_center)
        self.assertGreater(pose.rect_size[0], pose.rect_size[1])
        self.assertEqual(len(pose.quad), 4)
        self.assertAlmostEqual(pose.angle_deg, 0.0, delta=2.0)
        self.assertTrue(pose.is_straight)

    def test_small_angle_is_straight_with_default_threshold(self) -> None:
        frame = scanner._build_test_paper(3.0)
        analysis = scanner.CameraScanner().analyze_frame(frame)

        self.assertTrue(analysis.is_straight)
        self.assertEqual(analysis.status, scanner.ScanStatus.READY)
        self.assertEqual(analysis.pose.status_label, "已摆正")
        self.assertLessEqual(abs(analysis.angle), scanner.DEFAULT_TILT_THRESHOLD_DEG)

    def test_detect_paper_alignment_function(self) -> None:
        payload = scanner.detect_paper_alignment(
            scanner._build_test_paper(10.0),
            tilt_threshold=5.0,
        )
        self.assertFalse(payload["is_straight"])
        self.assertAlmostEqual(payload["angle"], -10.0, delta=2.0)
        self.assertEqual(payload["code_id"], "ALIEN-GARDEN-DEMO-001")

    def test_annotate_frame_draws_rotated_box_and_cjk_label(self) -> None:
        import numpy as np

        frame = scanner._build_test_paper(-12.0)
        analysis = scanner.CameraScanner().analyze_frame(frame, tilt_threshold=5.0)
        annotated = scanner.annotate_frame(frame, analysis.codes, analysis=analysis)

        self.assertEqual(annotated.shape, frame.shape)
        self.assertFalse(np.array_equal(annotated, frame))
        if scanner.PIL_AVAILABLE:
            canvas = np.full((80, 360, 3), 24, dtype=np.uint8)
            drawn = scanner._put_cjk_text(
                canvas,
                analysis.pose.status_label,
                (10, 10),
                (50, 70, 255),
                scale=0.9,
                backdrop=True,
            )
            self.assertTrue(drawn)
            self.assertGreater(int(np.count_nonzero(canvas != 24)), 0)


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
        from PyQt5.QtCore import QEvent, Qt
        from PyQt5.QtWidgets import QApplication

        cls.QEvent = QEvent
        cls.Qt = Qt
        cls.app = QApplication.instance() or QApplication([])
        cls.camera = scanner.CameraScanner()
        cls.level_frame = scanner._build_test_paper(0.0)
        cls.level_analysis = cls.camera.analyze_frame(cls.level_frame, tilt_threshold=8.0)
        cls.tilted_frame = scanner._build_test_paper(15.0)
        cls.tilted_analysis = cls.camera.analyze_frame(cls.tilted_frame, tilt_threshold=8.0)

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.window = showcase_main.StandbyWindow(
            video_path=None,
            windowed=True,
            force_fallback=True,
            camera_enabled=True,
            scene_seconds=8.0,
            capture_dir=Path(self._temp_dir.name),
        )
        self.window._play_shutter_feedback = lambda: None

    def tearDown(self) -> None:
        self.window.close()
        self._temp_dir.cleanup()

    def _press(self, key) -> None:
        from PyQt5.QtGui import QKeyEvent

        event = QKeyEvent(self.QEvent.KeyPress, key, self.Qt.NoModifier)
        self.window.keyPressEvent(event)

    def test_standby_video_or_idle_is_default(self) -> None:
        self.assertEqual(self.window.state, "standby")
        self.assertIs(self.window._stack.currentWidget(), self.window._idle_canvas)
        self.assertFalse(self.window._preview_mode)
        self.assertAlmostEqual(self.window._scene_seconds, 8.0, delta=0.1)

    def test_space_triggers_capture_and_scene(self) -> None:
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self._press(self.Qt.Key_Space)

        self.assertEqual(self.window._trigger_count, 1)
        self.assertEqual(self.window.state, "interactive")
        self.assertIs(self.window._stack.currentWidget(), self.window._scene_canvas)
        self.assertTrue(self.window._scene_timer.isActive())
        self.assertGreater(self.window._scene_timer.remainingTime(), 7000)

        self.assertIsNotNone(self.window.last_capture_path)
        assert self.window.last_capture_path is not None
        self.assertTrue(self.window.last_capture_path.is_file())
        self.assertEqual(self.window.last_capture_path.parent, Path(self._temp_dir.name))

        self.assertIsNotNone(self.window._scene_canvas.theme)
        assert self.window._scene_canvas.theme is not None
        self.assertEqual(self.window._scene_canvas.theme.key, "garden")

    def test_enter_triggers_capture(self) -> None:
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self._press(self.Qt.Key_Return)

        self.assertEqual(self.window._trigger_count, 1)
        self.assertEqual(self.window.state, "interactive")

    def test_tilted_paper_is_rejected_with_center_prompt(self) -> None:
        self.window._on_frame_ready(self.tilted_frame, self.tilted_analysis)
        self._press(self.Qt.Key_Space)

        self.assertEqual(self.window._trigger_count, 0)
        self.assertEqual(self.window.state, "standby")
        self.assertTrue(self.window._overlay.message_visible)
        self.assertIn("图纸未摆正", self.window._overlay.message)

    def test_missing_qr_is_rejected(self) -> None:
        analysis = scanner.FrameAnalysis(
            status=scanner.ScanStatus.READY,
            codes=(),
            pose=self.level_analysis.pose,
            frame_size=self.level_analysis.frame_size,
            timestamp=self.level_analysis.timestamp,
        )
        self.window._on_frame_ready(self.level_frame, analysis)
        self._press(self.Qt.Key_Enter)

        self.assertEqual(self.window._trigger_count, 0)
        self.assertIn("二维码", self.window._overlay.message)

    def test_finish_scene_fades_back_to_standby(self) -> None:
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self._press(self.Qt.Key_Space)
        self.assertEqual(self.window.state, "interactive")

        self.window._finish_scene()
        self.assertEqual(self.window.state, "reset")
        self.assertFalse(self.window._scene_timer.isActive())

        self.window._complete_reset()
        self.assertEqual(self.window.state, "standby")
        self.assertIs(self.window._stack.currentWidget(), self.window._idle_canvas)

    def test_repeated_key_while_interactive_does_not_capture_again(self) -> None:
        self.window._on_frame_ready(self.level_frame, self.level_analysis)
        self._press(self.Qt.Key_Space)
        self._press(self.Qt.Key_Space)

        self.assertEqual(self.window._trigger_count, 1)
        self.assertEqual(self.window.state, "interactive")

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
        self.assertFalse(args.camera_preview)

    def test_camera_preview_flag(self) -> None:
        args = showcase_main.parse_args(["--camera-preview"])
        self.assertTrue(args.camera_preview)

    def test_no_camera_flag_disables_scanner(self) -> None:
        args = showcase_main.parse_args(["--no-camera", "--windowed"])
        self.assertTrue(args.no_camera)
        self.assertTrue(args.windowed)

    def test_autostart_script_exists(self) -> None:
        script = PROJECT_DIR / "setup_autostart.bat"
        self.assertTrue(script.is_file())
        text = script.read_text(encoding="utf-8", errors="ignore")
        self.assertIn("Startup", text)
        self.assertIn("run.bat", text)

    def test_start_camera_flag_remains_compatible(self) -> None:
        args = showcase_main.parse_args(["--start-camera"])
        self.assertTrue(args.start_camera)
        self.assertFalse(args.no_camera)

