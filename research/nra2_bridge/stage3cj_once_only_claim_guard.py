"""S3CJ RESEARCH-ONLY: fail-closed S3CI sixth-EVAL once-only atomic claim.

This module does NOT run EVAL, load labels/weights, train, or alter LIVE.
Never call a real GitHub claim before an explicit one-time scoring approval.
The GitHub tag is a proposed durable research claim key, NOT a deployable
safeguard until the separate S3CI workflow is reviewed and integrated.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Callable, Mapping

REPO = "Kamara229/JessicaCore"
PR = 702
HEAD = "research/calibri2-nra2-s3ci-sixth128-once-only-20261115"
BASE = "research/calibri2-nra2-s3ch-sixth128-real43m-cpu-no-score-20261115"
TAG_REF = "refs/tags/research-nra2-s3ci-sixth128-eval-consumed-20261115"
ARM_TOKEN = "S3CI_EXPLICIT_ONCE_ONLY_APPROVAL"
SHA40 = re.compile(r"[0-9a-f]{40}\Z")


class ClaimDenied(RuntimeError):
    pass


def claim_once(
    env: Mapping[str, str],
    event: Mapping[str, object],
    post_ref: Callable[[str, str, str], int],
) -> str:
    """Atomically claim the sixth EVAL attempt; deny *all* non-201 outcomes.

    post_ref(repo, ref, head_sha) must perform one genuine GitHub POST git/refs.
    Unit tests inject a fake that never contacts GitHub. If the first attempt
    claims the tag then fails, another attempt is blocked (GitHub HTTP 422).
    """
    pr = event.get("pull_request", {})
    if not isinstance(pr, dict):
        raise ClaimDenied("missing pull_request object")
    base = pr.get("base", {})
    base = base if isinstance(base, dict) else {}
    base_repo = base.get("repo", {})
    base_repo = base_repo if isinstance(base_repo, dict) else {}
    head = pr.get("head", {})
    head = head if isinstance(head, dict) else {}
    head_repo = head.get("repo", {})
    head_repo = head_repo if isinstance(head_repo, dict) else {}
    sha = head.get("sha", "")
    if not (
        env.get("S3CI_ARM_TOKEN") == ARM_TOKEN
        and env.get("GITHUB_REPOSITORY") == REPO
        and env.get("GITHUB_EVENT_NAME") == "pull_request"
        and env.get("GITHUB_REF") == f"refs/pull/{PR}/merge"
        and env.get("GITHUB_BASE_REF") == BASE
        and env.get("GITHUB_HEAD_REF") == HEAD
        and env.get("GITHUB_RUN_ATTEMPT") == "1"
        and event.get("action") == "synchronize"
        and pr.get("number") == PR
        and base.get("ref") == BASE
        and base_repo.get("full_name") == REPO
        and head.get("ref") == HEAD
        and head_repo.get("full_name") == REPO
        and env.get("S3CI_HEAD_SHA") == sha
        and env.get("S3CI_APPROVED_HEAD_SHA") == sha
        and isinstance(sha, str)
        and SHA40.fullmatch(sha)
    ):
        raise ClaimDenied("unauthorized or unexpected S3CI workflow context")
    try:
        code = post_ref(REPO, TAG_REF, sha)
    except Exception as exc:
        raise ClaimDenied("atomic GitHub tag claim failed closed") from exc
    if code != 201:
        raise ClaimDenied("S3CI sixth EVAL already claimed or API failed: HTTP " + str(code))
    return TAG_REF


def gh_post_tag(repo: str, tag_ref: str, sha: str) -> int:
    """Execute only from a deliberately armed S3CI workflow; never in mock tests."""
    if repo != REPO or tag_ref != TAG_REF or not SHA40.fullmatch(sha):
        raise ClaimDenied("unexpected reference")
    body = json.dumps({"ref": tag_ref, "sha": sha})
    proc = subprocess.run(
        ["gh", "api", "-X", "POST", f"repos/{repo}/git/refs", "--input", "-",
         "--include"],
        input=body, text=True, capture_output=True, timeout=30,
        check=False,
    )
    # Fail closed even if stdout contains a stale/spoofed HTTP 201 status.
    if proc.returncode != 0:
        raise ClaimDenied("GitHub CLI failed: exit " + str(proc.returncode))
    # gh api --include prints the HTTP status line before response headers.
    statuses = re.findall(r"(?m)^HTTP/\S+\s+(\d{3})\b", proc.stdout)
    # No status or ambiguous status lines must never authorize one-time EVAL.
    if len(statuses) != 1:
        raise ClaimDenied("cannot uniquely verify GitHub atomic-create status")
    return int(statuses[0])


def main() -> None:
    import os
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path or not Path(event_path).is_file():
        raise ClaimDenied("GitHub pull_request event missing")
    event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    print("S3CJ_CLAIMED", claim_once(os.environ, event, gh_post_tag), flush=True)


if __name__ == "__main__":
    main()
