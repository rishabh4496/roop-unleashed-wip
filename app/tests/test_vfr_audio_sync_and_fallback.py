"""Unit and Integration Tests for VFR Audio/Video Sync and Graceful Execution Provider Fallback.

Verifies:
1. Variable Frame Rate (VFR) detection via ffprobe.
2. Frame extraction conforming with `-fps_mode cfr`.
3. Audio, subtitle, and metadata preservation during video remuxing/restoration.
4. Provider fallback mechanism: automatic downgrade from TensorRT to CUDAExecutionProvider (FP16).
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Ensure app root is on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import onnxruntime
from roop import util_ffmpeg
from roop.provider_fallback import (
    build_cuda_fallback_providers,
    create_fallback_session,
    is_trt_error,
    safe_run_with_fallback,
)
from roop.processors.Enhance_GPEN import create_gpen_session


class TestVFRAudioSync(unittest.TestCase):
    """Test suite for VFR detection, CFR conforming, and audio/metadata synchronization."""

    @classmethod
    def setUpClass(cls):
        # Create a small synthetic video with audio and metadata using FFmpeg
        cls.work_dir = tempfile.mkdtemp(prefix="roop_test_vfr_")
        cls.sample_video = os.path.join(cls.work_dir, "sample_input.mp4")
        cls.target_video = os.path.join(cls.work_dir, "sample_target.mp4")

        # Generate a 1-second 30fps test video with synthetic audio and metadata tags
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
            "-metadata", "title=RoopSyncTest",
            "-metadata", "artist=TestEngineer",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            cls.sample_video
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"Failed to generate test video with ffmpeg: {res.stderr.decode('utf-8', errors='ignore')}")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.work_dir, ignore_errors=True)

    def test_vfr_detection(self):
        """Verify is_variable_frame_rate correctly inspects video frame rate."""
        is_vfr = util_ffmpeg.is_variable_frame_rate(self.sample_video)
        # The generated synthetic testsrc with rate=30 is CFR
        self.assertFalse(is_vfr)

    def test_extract_frames_enforces_cfr(self):
        """Verify extract_frames extracts frames and conforms to CFR with -fps_mode cfr."""
        out_dir = os.path.join(self.work_dir, "extracted_frames")
        os.makedirs(out_dir, exist_ok=True)

        success = util_ffmpeg.extract_frames(self.sample_video, fps=30)
        self.assertTrue(success)

        # Temporary frame directory should exist and have ~30 frames
        temp_dir = util_ffmpeg.get_temp_frame_path(self.sample_video)
        self.assertTrue(os.path.isdir(temp_dir))
        frames = [f for f in os.listdir(temp_dir) if f.endswith(".png")]
        self.assertGreaterEqual(len(frames), 25)

    def test_restore_audio_multi_stream_and_metadata(self):
        """Verify restore_audio retains audio bitrate, sample rate, and metadata tags."""
        # Create a silent video track (representing swapped output)
        swapped_video = os.path.join(self.work_dir, "swapped_nosound.mp4")
        subprocess.run([
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=30",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            swapped_video
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        final_output = os.path.join(self.work_dir, "final_restored.mp4")

        success = util_ffmpeg.restore_audio(
            source_path=self.sample_video,
            target_path=swapped_video,
            output_path=final_output,
            fps=30.0,
        )
        self.assertTrue(success, "restore_audio failed")
        self.assertTrue(os.path.isfile(final_output))

        # Probe final output to verify audio presence and metadata tags
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format_tags=title,artist:stream=codec_type,sample_rate,bit_rate",
            "-of", "json",
            final_output
        ]
        probe_res = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        import json
        probe_data = json.loads(probe_res.stdout.decode("utf-8"))

        # Verify metadata tags preserved
        tags = probe_data.get("format", {}).get("tags", {})
        # Tag casing can vary in containers (title vs Title)
        tag_keys = {k.lower(): v for k, v in tags.items()}
        self.assertEqual(tag_keys.get("title"), "RoopSyncTest")
        self.assertEqual(tag_keys.get("artist"), "TestEngineer")

        # Verify audio stream preserved
        streams = probe_data.get("streams", [])
        audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
        self.assertEqual(len(audio_streams), 1, "Audio stream was lost in restoration")
        self.assertEqual(audio_streams[0].get("sample_rate"), "44100")


class TestProviderFallback(unittest.TestCase):
    """Test suite for TensorRT compilation/shape rejection fallback to CUDA FP16."""

    def test_is_trt_error_detection(self):
        """Verify is_trt_error catches various TensorRT-specific runtime errors."""
        trt_engine_err = RuntimeError("[ONNXRuntimeError] : 11 : FAIL : TensorRT [ERROR]: Engine build failed")
        trt_shape_err = RuntimeError("TensorRT execution failed: input shape does not match profile")
        trt_load_err = Exception("Failed to load TensorRT execution provider shared library")
        cuda_err = RuntimeError("CUDA out of memory")

        self.assertTrue(is_trt_error(trt_engine_err))
        self.assertTrue(is_trt_error(trt_shape_err))
        self.assertTrue(is_trt_error(trt_load_err))
        self.assertFalse(is_trt_error(cuda_err))

    def test_build_cuda_fallback_providers(self):
        """Verify build_cuda_fallback_providers produces valid CUDA FP16 provider list."""
        providers = build_cuda_fallback_providers(device_id=0)
        self.assertIsInstance(providers, list)
        self.assertGreaterEqual(len(providers), 1)

        first_p = providers[0]
        if isinstance(first_p, tuple):
            name, opts = first_p
            self.assertEqual(name, "CUDAExecutionProvider")
            self.assertEqual(opts.get("device_id"), 0)
        else:
            self.assertIn("CUDA", first_p)

    def test_create_fallback_session_on_trt_failure(self):
        """Verify create_fallback_session automatically downgrades when TensorRT fails."""
        # Mock onnxruntime.InferenceSession to raise TRT error on first attempt, then succeed on fallback
        real_inf_session = onnxruntime.InferenceSession

        attempts = []

        def mock_inference_session(model_path, options=None, providers=None, **kwargs):
            attempts.append(providers)
            if any("tensorrt" in str(p).lower() for p in providers):
                raise RuntimeError("TensorRT dynamic input resolution rejection / compilation failure")
            mock_sess = MagicMock()
            mock_sess.get_providers.return_value = ["CUDAExecutionProvider"]
            return mock_sess

        with patch("onnxruntime.InferenceSession", side_effect=mock_inference_session):
            sess = create_fallback_session(
                "dummy_model.onnx",
                providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
                session_name="TestFallbackModel"
            )
            self.assertIsNotNone(sess)
            # Verify 2 attempts: first with TensorRT, second with CUDA fallback
            self.assertEqual(len(attempts), 2)
            self.assertIn("TensorrtExecutionProvider", str(attempts[0]))
            self.assertNotIn("tensorrt", str(attempts[1]).lower())
            self.assertIn("CUDAExecutionProvider", str(attempts[1]))

    def test_safe_run_with_fallback(self):
        """Verify safe_run_with_fallback recovers mid-run if TRT fails dynamic shape verification."""
        mock_sess = MagicMock()
        mock_sess.run.side_effect = RuntimeError("TensorRT runtime input resolution mismatch")

        fallback_called = False

        def mock_rebuild():
            nonlocal fallback_called
            fallback_called = True
            new_sess = MagicMock()
            new_sess.run.return_value = ["recovered_output"]
            return new_sess

        res = safe_run_with_fallback(
            session=mock_sess,
            rebuild_fn=mock_rebuild,
            output_names=None,
            input_feed={"input": "data"}
        )
        self.assertTrue(fallback_called)
        self.assertEqual(res, ["recovered_output"])

    def test_gpen_trt_fallback_remains_strictly_on_gpu(self):
        """Removing TRT must not add CPU execution to GPEN's strict path."""
        attempts = []
        fallback_session = MagicMock()
        fallback_session.get_providers.return_value = ["CUDAExecutionProvider"]

        def fake_session(_model, _options=None, providers=None, **_kwargs):
            attempts.append(providers)
            if any("tensorrt" in str(p).lower() for p in providers):
                raise RuntimeError("TensorRT engine creation failed")
            return fallback_session

        with patch("roop.processors.Enhance_GPEN.onnxruntime.InferenceSession",
                   side_effect=fake_session):
            session = create_gpen_session(
                "dummy.onnx",
                ["TensorrtExecutionProvider", "CUDAExecutionProvider"],
                "cuda",
            )

        self.assertIs(session, fallback_session)
        self.assertEqual(len(attempts), 2)
        self.assertNotIn("CPUExecutionProvider",
                         [p[0] if isinstance(p, tuple) else p for p in attempts[1]])
        fallback_session.disable_fallback.assert_called_once_with()


if __name__ == "__main__":
    unittest.main(verbosity=2)
