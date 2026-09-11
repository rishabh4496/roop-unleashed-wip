"""A running API job must not observe later cross-tab gallery mutations."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import api  # noqa: E402
from roop.ProcessEntry import ProcessEntry  # noqa: E402


class TestJobStateSnapshot(unittest.TestCase):
    def setUp(self):
        self.old_files = list(api.list_files_process)
        self.old_sources = list(api.roop_globals.INPUT_FACESETS)
        self.old_targets = list(api.roop_globals.TARGET_FACES)
        self.old_groups = list(api.roop_globals.TARGET_FACE_GROUP)
        self.old_selected = api.state.selected_input_face_index

    def tearDown(self):
        api.list_files_process[:] = self.old_files
        api.roop_globals.INPUT_FACESETS[:] = self.old_sources
        api.roop_globals.TARGET_FACES[:] = self.old_targets
        api.roop_globals.TARGET_FACE_GROUP[:] = self.old_groups
        api.state.selected_input_face_index = self.old_selected

    def test_snapshot_copies_lists_and_process_entries(self):
        entry = ProcessEntry('target.mp4', 10, 50, 30000 / 1001)
        entry.total_frames = 100
        source = object()
        target = object()
        api.list_files_process[:] = [entry]
        api.roop_globals.INPUT_FACESETS[:] = [source]
        api.roop_globals.TARGET_FACES[:] = [target]
        api.roop_globals.TARGET_FACE_GROUP[:] = [7]
        api.state.selected_input_face_index = 3

        snapshot = api._snapshot_job_state()

        entry.startframe = 99
        api.list_files_process.clear()
        api.roop_globals.INPUT_FACESETS.clear()
        api.roop_globals.TARGET_FACES.clear()
        api.roop_globals.TARGET_FACE_GROUP.clear()
        api.state.selected_input_face_index = 0

        self.assertEqual(snapshot['files'][0].startframe, 10)
        self.assertEqual(snapshot['files'][0].total_frames, 100)
        self.assertIs(snapshot['input_facesets'][0], source)
        self.assertIs(snapshot['target_faces'][0], target)
        self.assertEqual(snapshot['target_face_groups'], [7])
        self.assertEqual(snapshot['selected_input_face_index'], 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
