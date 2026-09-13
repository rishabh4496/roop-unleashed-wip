import os
import tempfile
import unittest
from unittest import mock


class OfflinePolicyTests(unittest.TestCase):
    def setUp(self):
        from roop import offline

        self.offline = offline
        self.env = {
            key: os.environ.get(key)
            for key in (
                "ROOP_OFFLINE",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
                "HF_DATASETS_OFFLINE",
                "GRADIO_ANALYTICS_ENABLED",
                "GRADIO_TELEMETRY_ENABLED",
                "NO_ALBUMENTATIONS_UPDATE",
                "HF_HUB_DISABLE_TELEMETRY",
                "HF_HUB_ETAG_TIMEOUT",
                "HF_HUB_DOWNLOAD_TIMEOUT",
            )
        }

    def tearDown(self):
        for key, value in self.env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_explicit_flag_sets_all_offline_and_telemetry_flags(self):
        with mock.patch.dict(os.environ, {
            "ROOP_OFFLINE": "",
            "HF_HUB_OFFLINE": "",
            "TRANSFORMERS_OFFLINE": "",
            "HF_DATASETS_OFFLINE": "",
        }, clear=False):
            self.assertTrue(self.offline.configure_startup_environment(
                ["run.py", "--offline"]))
            self.assertEqual(os.environ["ROOP_OFFLINE"], "1")
            for key in self.offline.OFFLINE_ENV_VARS:
                self.assertEqual(os.environ[key], "1")
            self.assertEqual(os.environ["GRADIO_ANALYTICS_ENABLED"], "False")
            self.assertEqual(os.environ["GRADIO_TELEMETRY_ENABLED"], "False")
            self.assertEqual(os.environ["NO_ALBUMENTATIONS_UPDATE"], "1")
            self.assertEqual(os.environ["HF_HUB_DISABLE_TELEMETRY"], "1")
            self.assertEqual(os.environ["HF_HUB_ETAG_TIMEOUT"], "3")
            self.assertEqual(os.environ["HF_HUB_DOWNLOAD_TIMEOUT"], "3")

    def test_dns_failure_automatically_enters_offline_mode(self):
        with mock.patch.dict(os.environ, {
            "ROOP_OFFLINE": "",
            "HF_HUB_OFFLINE": "",
            "TRANSFORMERS_OFFLINE": "",
            "HF_DATASETS_OFFLINE": "",
        }, clear=False), mock.patch.object(
            self.offline.socket, "getaddrinfo", side_effect=OSError("no DNS")):
            self.assertTrue(self.offline.configure_startup_environment(["run.py"]))
            self.assertEqual(os.environ["ROOP_OFFLINE"], "1")

    def test_offline_download_fails_without_opening_url(self):
        from roop import utilities

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"ROOP_OFFLINE": "1"}, clear=False), mock.patch.object(
                utilities.urllib.request, "urlopen") as urlopen:
            with self.assertRaises(utilities.OfflineModelError) as error:
                utilities.conditional_download(
                    directory,
                    ["https://example.invalid/model.onnx"],
                )
            self.assertIn("model.onnx", str(error.exception))
            urlopen.assert_not_called()

    def test_existing_local_file_wins_without_network(self):
        from roop import utilities

        with tempfile.TemporaryDirectory() as directory:
            local_file = os.path.join(directory, "model.onnx")
            with open(local_file, "wb") as stream:
                stream.write(b"cached")
            with mock.patch.object(utilities.urllib.request, "urlopen") as urlopen:
                utilities.conditional_download(
                    directory,
                    ["https://example.invalid/model.onnx"],
                )
                urlopen.assert_not_called()

    def test_runtime_lock_blocks_remote_downloads(self):
        from roop import utilities

        old_locked = utilities._RUNTIME_DOWNLOADS_LOCKED
        try:
            utilities._RUNTIME_DOWNLOADS_LOCKED = False
            with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                os.environ, {"ROOP_OFFLINE": ""}, clear=False), mock.patch.object(
                    utilities.urllib.request, "urlopen") as urlopen:
                utilities.lock_runtime_downloads()
                with self.assertRaises(utilities.OfflineModelError):
                    utilities.conditional_download(
                        directory,
                        ["https://example.invalid/model.onnx"],
                    )
                urlopen.assert_not_called()
        finally:
            utilities._RUNTIME_DOWNLOADS_LOCKED = old_locked


if __name__ == "__main__":
    unittest.main()
