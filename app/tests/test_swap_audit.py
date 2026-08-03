"""Swap audit: the breakdown that says WHY a detected face was not swapped.

The audit exists to answer one report — "the swap flickers on and off" — which
every identity gate in swap_faces can produce and which all look identical in
the output: a frame where a face was found but nothing was painted. Its whole
value is pointing at the right gate.

The original version bucketed by matching the veto MESSAGE, which this test
caught as unsound: two of the four messages open with nearly the same words
("face is 0.91 from its assigned person" / "face is 1.23 from the selected
person") and the token separating them sits at the tail of a two-part implicit
concatenation. So the buckets are now named explicitly at each veto site, and
what needs guarding is that every site still names one.
"""
import ast
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from roop.procmgr_runtime import (          # noqa: E402
    VETO_BUCKETS, _audit, _audit_hit, _audit_report, _audit_reset,
)

_PROCMGR = os.path.join(os.path.dirname(__file__), '..', 'roop', 'ProcessMgr.py')


def _swap_faces_body():
    with open(_PROCMGR, encoding='utf-8') as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == 'swap_faces':
            return node
    raise AssertionError('swap_faces not found in ProcessMgr.py')


class TestAuditBuckets(unittest.TestCase):
    def setUp(self):
        _audit_reset()

    def test_every_veto_assignment_is_paired_with_a_kind(self):
        """`veto = ...` and `veto_kind = ...` must come in pairs.

        A veto added without its bucket would refuse faces that never appear in
        the audit — the report would look clean while the swap flickered, which
        is the exact failure the audit exists to prevent. Counted structurally
        rather than by reading the strings, so rewording a message is free.
        """
        body = _swap_faces_body()
        n_veto = n_kind = 0
        for node in ast.walk(body):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                t = node.targets[0]
                if isinstance(t, ast.Name):
                    if t.id == 'veto':
                        # Skip the `veto = None` initialiser.
                        if not (isinstance(node.value, ast.Constant)
                                and node.value.value is None):
                            n_veto += 1
                    elif t.id == 'veto_kind':
                        if not (isinstance(node.value, ast.Constant)
                                and node.value.value is None):
                            n_kind += 1
        self.assertGreaterEqual(n_veto, 4, 'expected at least 4 veto reasons')
        self.assertEqual(n_veto, n_kind,
                         f'{n_veto} veto assignments but {n_kind} veto_kind '
                         f'assignments — a refusal would go uncounted')

    def test_veto_kinds_used_are_all_known_buckets(self):
        """Every bucket named in swap_faces must exist in procmgr_runtime."""
        body = _swap_faces_body()
        used = set()
        for node in ast.walk(body):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'veto_kind'
                    and isinstance(node.value, ast.Name)):
                used.add(node.value.id)
        self.assertEqual(len(used), 4, f'expected 4 distinct buckets, got {used}')
        known = {'VETO_SOURCE_REUSED', 'VETO_SINGLE_ABS',
                 'VETO_OTHER_FITS', 'VETO_FAR_FROM_OWN'}
        self.assertEqual(used, known)

    def test_bucket_names_are_distinct(self):
        # Two buckets sharing a name would silently merge two gates in the report.
        self.assertEqual(len(set(VETO_BUCKETS)), len(VETO_BUCKETS))

    def test_report_is_silent_when_track_mode_never_ran(self):
        # No 'faces seen' -> nothing was gated -> must not print, or every image
        # swap would emit an empty audit block.
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _audit_report()
        self.assertEqual(buf.getvalue(), '')

    def test_report_counts_unswapped_faces(self):
        import contextlib
        import io
        _audit_hit('faces seen', 10)
        _audit_hit('swapped (identity lock)', 6)
        _audit_hit('swapped (per-frame match)', 1)
        _audit_hit(VETO_BUCKETS[1], 3)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _audit_report()
        out = buf.getvalue()
        self.assertIn('SWAP AUDIT', out)
        self.assertIn('3 of 10 detected faces', out)   # 10 seen, 7 swapped
        self.assertIn(VETO_BUCKETS[1], out)

    def test_report_says_nothing_missed_when_all_swapped(self):
        import contextlib
        import io
        _audit_hit('faces seen', 4)
        _audit_hit('swapped (identity lock)', 4)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            _audit_report()
        self.assertNotIn('were NOT swapped', buf.getvalue())

    def test_reset_clears_between_jobs(self):
        _audit_hit('faces seen', 5)
        _audit_reset()
        self.assertEqual(dict(_audit), {})

    def test_reset_and_report_share_one_scope(self):
        """The counters must be cleared once per CLIP, in the same function that
        reports them.

        This was wrong on the first cut: the reset sat in `initialize()`, which
        core.py calls once before handing batch_process a whole LIST of files,
        while the report runs at the end of each file. Every clip after the first
        therefore reported its own counts plus all previous clips' — the exact
        confusion the audit exists to remove, and invisible on a single-clip run.
        """
        with open(_PROCMGR, encoding='utf-8') as fh:
            tree = ast.parse(fh.read())
        where = {'_audit_reset': set(), '_audit_report': set()}
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for call in ast.walk(node):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
                        and call.func.id in where):
                    where[call.func.id].add(node.name)
        self.assertEqual(where['_audit_reset'], where['_audit_report'],
                         f'reset runs in {where["_audit_reset"]} but report runs '
                         f'in {where["_audit_report"]} — counters would span clips')
        self.assertEqual(len(where['_audit_report']), 1,
                         f'expected exactly one reporting scope, got {where}')


if __name__ == '__main__':
    unittest.main()
