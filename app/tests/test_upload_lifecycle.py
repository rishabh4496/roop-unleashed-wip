"""Upload filenames, deletion guards, and source-gallery updates are race safe."""

import io
import os
import sys
import tempfile
import threading
import unittest

from fastapi import UploadFile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api_media  # noqa: E402
import source_gallery  # noqa: E402


class TestUploadLifecycle(unittest.TestCase):
    def test_same_name_uploads_are_atomically_unique(self):
        old_root = api_media.API_TEMP
        try:
            with tempfile.TemporaryDirectory() as root:
                api_media.API_TEMP = root
                paths = []
                lock = threading.Lock()

                def upload(index):
                    item = UploadFile(
                        filename='same image.png', file=io.BytesIO(str(index).encode()))
                    path = api_media._save_upload(item)
                    with lock:
                        paths.append(path)

                workers = [threading.Thread(target=upload, args=(i,))
                           for i in range(16)]
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(timeout=5.0)

                self.assertEqual(len(paths), 16)
                self.assertEqual(len(set(paths)), 16)
                self.assertTrue(all(os.path.isfile(path) for path in paths))
                self.assertTrue(all(api_media._delete_upload(path) for path in paths))
        finally:
            api_media.API_TEMP = old_root

    def test_delete_refuses_paths_outside_upload_root(self):
        old_root = api_media.API_TEMP
        try:
            with tempfile.TemporaryDirectory() as root, \
                 tempfile.NamedTemporaryFile(delete=False) as outside:
                api_media.API_TEMP = root
                outside_path = outside.name
            self.assertFalse(api_media._delete_upload(outside_path))
            self.assertTrue(os.path.isfile(outside_path))
            os.remove(outside_path)
        finally:
            api_media.API_TEMP = old_root


class TestSourceGalleryLock(unittest.TestCase):
    def setUp(self):
        self.faces = list(source_gallery.roop_globals.INPUT_FACESETS)
        self.thumbs = list(source_gallery.ui_globals.ui_input_thumbs)
        source_gallery._sources_clear()

    def tearDown(self):
        source_gallery.roop_globals.INPUT_FACESETS[:] = self.faces
        source_gallery.ui_globals.ui_input_thumbs[:] = self.thumbs

    def test_parallel_appends_keep_faces_and_thumbnails_paired(self):
        workers = [threading.Thread(
            target=source_gallery._sources_append, args=(f'face-{i}', f'thumb-{i}'))
            for i in range(64)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=5.0)
        pairs = list(zip(source_gallery.roop_globals.INPUT_FACESETS,
                         source_gallery.ui_globals.ui_input_thumbs))
        self.assertEqual(len(pairs), 64)
        self.assertTrue(all(face.removeprefix('face-') == thumb.removeprefix('thumb-')
                            for face, thumb in pairs))


if __name__ == '__main__':
    unittest.main(verbosity=2)
