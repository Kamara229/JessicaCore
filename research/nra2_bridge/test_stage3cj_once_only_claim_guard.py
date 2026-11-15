"""Offline-only negative controls: NO GitHub, EVAL128, or Jessica calls."""
import unittest

from research.nra2_bridge.stage3cj_once_only_claim_guard import (
    ARM_TOKEN, HEAD, PR, REPO, ClaimDenied, claim_once,
)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.env = {
            "S3CI_ARM_TOKEN": ARM_TOKEN,
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_RUN_ATTEMPT": "1",
            "S3CI_HEAD_SHA": self.sha,
        }
        self.event = {
            "action": "synchronize", "pull_request": {
                "number": PR, "head": {
                    "ref": HEAD, "sha": self.sha,
                    "repo": {"full_name": REPO}
                }
            }
        }

    def test_first_claim_wins_even_if_attempt_later_fails(self):
        claims = set()
        def fake_post(repo, ref, sha):
            key = (repo, ref)
            if key in claims:
                return 422
            claims.add(key)
            return 201
        self.assertTrue(claim_once(self.env, self.event, fake_post).startswith("refs/tags/"))
        # Simulated scoring process crashes AFTER acquiring the tag.
        with self.assertRaises(ClaimDenied):
            claim_once(self.env, self.event, fake_post)
        self.assertEqual(len(claims), 1)

    def test_two_concurrent_claims_only_one_wins(self):
        """Mocked atomic GitHub-create: simultaneous claims cannot both authorize EVAL."""
        from concurrent.futures import ThreadPoolExecutor
        from threading import Barrier, Lock

        barrier = Barrier(2)
        lock = Lock()
        claimed = set()

        def atomic_mock(repo, ref, sha):
            barrier.wait(timeout=5)
            with lock:
                key = (repo, ref)
                if key in claimed:
                    return 422
                claimed.add(key)
                return 201

        def attempt():
            try:
                claim_once(self.env, self.event, atomic_mock)
            except ClaimDenied:
                return "blocked"
            return "claimed"

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: attempt(), range(2)))
        self.assertEqual(sorted(results), ["blocked", "claimed"])
        self.assertEqual(len(claimed), 1)

    def test_api_failure_denied(self):
        with self.assertRaises(ClaimDenied):
            claim_once(self.env, self.event, lambda *_: 500)
        with self.assertRaises(ClaimDenied):
            claim_once(self.env, self.event, lambda *_: (_ for _ in ()).throw(OSError("offline")))

    def test_missing_approval_denied_before_api(self):
        env = dict(self.env)
        env.pop("S3CI_ARM_TOKEN")
        def never_call(*_):
            raise AssertionError("API should not be called")
        with self.assertRaises(ClaimDenied):
            claim_once(env, self.event, never_call)

    def test_rerun_denied_before_api(self):
        env = dict(self.env, GITHUB_RUN_ATTEMPT="2")
        with self.assertRaises(ClaimDenied):
            claim_once(env, self.event, lambda *_: 201)

    def test_wrong_pr_or_branch_denied(self):
        bad = {"action": "synchronize", "pull_request": {
            "number": PR + 1, "head": self.event["pull_request"]["head"]
        }}
        with self.assertRaises(ClaimDenied):
            claim_once(self.env, bad, lambda *_: 201)

    def test_scoring_job_hard_stop_and_pre_eval_claim_order(self):
        """Static workflow guard: scoring cannot run before separate authorization."""
        from pathlib import Path
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" /
                    "nra2-stage3ci-sixth128-once-only-paired-role-reach.yml")
        text = workflow.read_text(encoding="utf-8")
        conditions = [line.strip() for line in text.splitlines()
                      if line.startswith("    if: ")]
        self.assertEqual(len(conditions), 1)
        self.assertTrue(conditions[0].endswith(" && false"))
        self.assertIn("permissions:\n  contents: read", text)
        claim = text.index("- name: S3CJ atomic sixth-EVAL claim")
        first_data = text.index("- name: Load SEALED sixth S3CG")
        self.assertLess(claim, first_data)
        self.assertLess(first_data, text.index("- name: Load genuinely frozen original43M CPU"))


    def test_pr702_synchronize_mock_never_starts_scoring(self):
        """Offline model of PR #702 synchronization, with actual workflow guards."""
        from pathlib import Path
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" /
                    "nra2-stage3ci-sixth128-once-only-paired-role-reach.yml")
        source = workflow.read_text(encoding="utf-8")
        base = "research/calibri2-nra2-s3ch-sixth128-real43m-cpu-no-score-20261115"
        self.assertIn("types: [synchronize]", source)
        self.assertIn("      - " + base, source)
        mock_event = {"action": "synchronize", "pull_request": {
            "number": 702, "base": {"ref": base},
            "head": {"ref": HEAD, "repo": {"full_name": REPO}},
        }}
        event_allowed = (mock_event["action"] == "synchronize" and
                         mock_event["pull_request"]["base"]["ref"] == base)
        self.assertTrue(event_allowed)
        lines = [line.strip() for line in source.splitlines()
                 if line.startswith("    if: ")]
        self.assertEqual(len(lines), 1)
        job_condition = lines[0]
        expected = ("if: github.event.pull_request.head.repo.full_name == github.repository"
                    " && github.event.pull_request.head.ref == '" + HEAD + "' && false")
        self.assertEqual(job_condition, expected)
        # The source condition is deliberately FALSE, even on a matching PR #702 sync.
        head_matches = (mock_event["pull_request"]["head"]["ref"] == HEAD and
                        mock_event["pull_request"]["head"]["repo"]["full_name"] == REPO)
        self.assertTrue(head_matches)
        self.assertFalse(event_allowed and head_matches and job_condition.endswith(" && false") is False)
        self.assertIn("permissions:\\n  contents: read", source)


if __name__ == "__main__":
    unittest.main()
