"""Offline-only negative controls: NO GitHub, EVAL128, or Jessica calls."""
import unittest

from research.nra2_bridge.stage3cj_once_only_claim_guard import (
    ARM_TOKEN, BASE, HEAD, PR, REPO, ClaimDenied, claim_once,
)


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.sha = "a" * 40
        self.env = {
            "S3CI_ARM_TOKEN": ARM_TOKEN,
            "GITHUB_REPOSITORY": REPO,
            "GITHUB_EVENT_NAME": "pull_request",
            "GITHUB_REF": f"refs/pull/{PR}/merge",
            "GITHUB_BASE_REF": BASE,
            "GITHUB_HEAD_REF": HEAD,
            "GITHUB_RUN_ATTEMPT": "1",
            "S3CI_HEAD_SHA": self.sha,
            "S3CI_APPROVED_HEAD_SHA": self.sha,
        }
        self.event = {
            "action": "synchronize", "pull_request": {
                "number": PR, "base": {
                    "ref": BASE, "repo": {"full_name": REPO}
                }, "head": {
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


    def test_unpinned_or_mismatched_approved_sha_denied_before_api(self):
        """Only explicitly SHA-pinned approved scoring context may reach GitHub."""
        def never_call(*_):
            raise AssertionError("GitHub API must not be called for SHA mismatch")
        for invalid in ("", "b" * 40, "a" * 39, "not-a-sha"):
            with self.subTest(approved_sha=invalid):
                with self.assertRaises(ClaimDenied):
                    claim_once(dict(self.env, S3CI_APPROVED_HEAD_SHA=invalid), self.event, never_call)
        omitted = dict(self.env)
        omitted.pop("S3CI_APPROVED_HEAD_SHA")
        with self.assertRaises(ClaimDenied):
            claim_once(omitted, self.event, never_call)

    def test_pull_request_merge_ref_head_base_env_fail_closed(self):
        """Offline: reject other PR refs and mismatched runner head/base context."""
        def never_call(*_):
            raise AssertionError("GitHub API must not be called")
        cases = (
            ("GITHUB_REF", "refs/pull/703/merge"),
            ("GITHUB_REF", f"refs/heads/{HEAD}"),
            ("GITHUB_BASE_REF", "main"),
            ("GITHUB_HEAD_REF", "unexpected-head"),
            ("GITHUB_REF", ""),
        )
        for key, value in cases:
            with self.subTest(key=key, value=value):
                with self.assertRaises(ClaimDenied):
                    claim_once(dict(self.env, **{key: value}), self.event, never_call)

    def test_wrong_base_branch_or_repository_denied_before_api(self):
        """Offline: only the intended S3CH base may claim the one-time tag."""
        def never_call(*_):
            raise AssertionError("GitHub API must not be called for wrong base")
        for malformed in (
            {"ref": BASE + "-wrong", "repo": {"full_name": REPO}},
            {"ref": BASE, "repo": {"full_name": "another/repo"}},
            {"ref": BASE},
            {},
        ):
            with self.subTest(malformed=malformed):
                event = dict(self.event)
                event["pull_request"] = dict(self.event["pull_request"], base=malformed)
                with self.assertRaises(ClaimDenied):
                    claim_once(self.env, event, never_call)

    def test_wrong_pr_or_branch_denied(self):
        bad = {"action": "synchronize", "pull_request": {
            "number": PR + 1, "head": self.event["pull_request"]["head"]
        }}
        with self.assertRaises(ClaimDenied):
            claim_once(self.env, bad, lambda *_: 201)

    def test_scoring_workflow_has_only_synchronize_trigger_and_one_job(self):
        """Offline static check: no hidden extra trigger or unguarded scoring job."""
        from pathlib import Path
        workflow = (Path(__file__).resolve().parents[2] / ".github" / "workflows" /
                    "nra2-stage3ci-sixth128-once-only-paired-role-reach.yml")
        source = workflow.read_text(encoding="utf-8")
        triggers = source.split("'on':\\n", 1)[1].split("\\npermissions:", 1)[0]
        trigger_keys = [line for line in triggers.splitlines()
                        if line.startswith("  ") and not line.startswith("    ")
                        and line.endswith(":")]
        self.assertEqual(trigger_keys, ["  pull_request:"])
        self.assertIn("    types: [synchronize]", triggers)
        self.assertIn("    branches:\\n      - " + BASE, triggers)
        self.assertEqual(triggers.count("      - " + BASE), 1)
        jobs = source.split("\\njobs:\\n", 1)[1]
        job_keys = [line for line in jobs.splitlines()
                    if line.startswith("  ") and not line.startswith("    ")
                    and line.endswith(":")]
        self.assertEqual(job_keys, ["  s3ci_one_time_SIXTH_s3cg128_new_joint_semantic_roles:"])
        self.assertIn("  cancel-in-progress: false", source)

    def test_workflow_permissions_and_manual_arm_are_fail_closed(self):
        """Offline-only: neither workflow can write or arm EVAL by default."""
        from pathlib import Path
        workflows = Path(__file__).resolve().parents[2] / ".github" / "workflows"
        scorer = (workflows / "nra2-stage3ci-sixth128-once-only-paired-role-reach.yml").read_text(encoding="utf-8")
        unit = (workflows / "nra2-stage3cj-once-only-guard-offline-unit.yml").read_text(encoding="utf-8")
        for label, source in (("scorer", scorer), ("unit", unit)):
            with self.subTest(workflow=label):
                self.assertIn("permissions:\n  contents: read", source)
                self.assertNotIn("contents: write", source)
                self.assertNotIn("pull_request_target:", source)
                self.assertNotIn("workflow_dispatch:", source)
                self.assertNotIn("repository_dispatch:", source)
        self.assertIn("S3CI_ARM_TOKEN: ${{ vars.NRA2_S3CI_SIXTH128_APPROVAL }}", scorer)
        self.assertIn("S3CI_APPROVED_HEAD_SHA: ${{ vars.NRA2_S3CI_APPROVED_HEAD_SHA }}", scorer)
        self.assertNotIn("S3CI_APPROVED_HEAD_SHA: ${{ github.event.pull_request.head.sha }}", scorer)
        self.assertIn(" && false", scorer)
        self.assertNotIn("actions/download-artifact", unit)

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
        self.assertIn("permissions:" + chr(10) + "  contents: read", source)


    def test_gh_cli_nonzero_exit_fail_closed_even_if_stdout_says_201(self):
        """No network: mock gh process; CLI exit failure must block claim."""
        from subprocess import CompletedProcess
        from unittest.mock import patch
        from research.nra2_bridge.stage3cj_once_only_claim_guard import gh_post_tag, TAG_REF

        with patch("research.nra2_bridge.stage3cj_once_only_claim_guard.subprocess.run",
                   return_value=CompletedProcess(["gh"], 1, "HTTP/2 201 Created\n", "error")) as mocked:
            with self.assertRaises(ClaimDenied):
                claim_once(self.env, self.event, gh_post_tag)
            mocked.assert_called_once()
            command = mocked.call_args.args[0]
            self.assertEqual(command[:4], ["gh", "api", "-X", "POST"])
            self.assertIn(TAG_REF, mocked.call_args.kwargs["input"])

    def test_gh_cli_http_201_success_only_with_zero_exit(self):
        """No network: a single mocked success can claim, without EVAL loading."""
        from subprocess import CompletedProcess
        from unittest.mock import patch
        from research.nra2_bridge.stage3cj_once_only_claim_guard import gh_post_tag

        with patch("research.nra2_bridge.stage3cj_once_only_claim_guard.subprocess.run",
                   return_value=CompletedProcess(["gh"], 0, "HTTP/2 201 Created\n", "")) as mocked:
            self.assertTrue(claim_once(self.env, self.event, gh_post_tag).startswith("refs/tags/"))
            mocked.assert_called_once()


    def test_gh_cli_ambiguous_or_missing_status_denied_offline(self):
        """Mock-only: even zero-exit CLI cannot authorize ambiguous HTTP status."""
        from subprocess import CompletedProcess
        from unittest.mock import patch
        from research.nra2_bridge.stage3cj_once_only_claim_guard import gh_post_tag

        cases = (
            "HTTP/2 500 Error\nHTTP/2 201 Created\n",
            "HTTP/2 201 Created\nHTTP/2 422 Unprocessable Entity\n",
            "HTTP 201 Created\n",
            "not an HTTP response",
        )
        for stdout in cases:
            with self.subTest(stdout=stdout):
                with patch("research.nra2_bridge.stage3cj_once_only_claim_guard.subprocess.run",
                           return_value=CompletedProcess(["gh"], 0, stdout, "")) as mocked:
                    with self.assertRaises(ClaimDenied):
                        claim_once(self.env, self.event, gh_post_tag)
                    mocked.assert_called_once()

    def test_gh_cli_http_422_and_500_denied_offline(self):
        """Mock-only: no EVAL on already-claimed or server-failure responses."""
        from subprocess import CompletedProcess
        from unittest.mock import patch
        from research.nra2_bridge.stage3cj_once_only_claim_guard import gh_post_tag

        for code in (422, 500):
            with self.subTest(code=code):
                with patch("research.nra2_bridge.stage3cj_once_only_claim_guard.subprocess.run",
                           return_value=CompletedProcess(["gh"], 0, f"HTTP/2 {code} Error\n", "")) as mocked:
                    with self.assertRaises(ClaimDenied):
                        claim_once(self.env, self.event, gh_post_tag)
                    mocked.assert_called_once()


if __name__ == "__main__":
    unittest.main()
