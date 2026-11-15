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


if __name__ == "__main__":
    unittest.main()
