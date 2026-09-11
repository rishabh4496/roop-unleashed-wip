"""Face detector errors must not masquerade as legitimate no-face results."""

import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roop import face_util  # noqa: E402


class TestFaceDetectionErrors(unittest.TestCase):
    def test_invalid_frame_is_a_safe_empty_result(self):
        self.assertEqual(face_util.get_all_faces(None), [])
        self.assertEqual(
            face_util.get_all_faces(np.empty((0, 0, 3), dtype=np.uint8)), [])

    def test_provider_failure_is_propagated(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        with mock.patch.object(
                face_util, '_detect_faces', side_effect=ValueError('bad binding')):
            with self.assertRaisesRegex(RuntimeError, 'Face detector failed'):
                face_util.get_all_faces(frame)

    def test_first_face_is_leftmost(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        right = SimpleNamespace(bbox=np.array([10, 0, 20, 10]))
        left = SimpleNamespace(bbox=np.array([1, 0, 8, 10]))
        with mock.patch.object(face_util, '_detect_faces', return_value=[right, left]), \
             mock.patch.object(face_util, '_enrich_detected_faces',
                               side_effect=lambda _frame, faces: faces):
            self.assertIs(face_util.get_first_face(frame), left)


if __name__ == '__main__':
    unittest.main(verbosity=2)
