"""S144 read-only S142 FP16 source preflight; no training, inference or deployment.

Verifies the GitHub artifact ZIP, checkpoint digest and shape, and the original
S142 physical guard. Full 456+128 CPU replay is attested, NOT rerun here.
"""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import urllib.parse
import urllib.request
import zipfile


class SourcePinError(ValueError):
    pass


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise SourcePinError(reason)


def verify_archive(blob: bytes, manifest: dict) -> dict:
    require(hashlib.sha256(blob).hexdigest() == manifest["source_zip_sha256"],
            "Source ZIP digest mismatch")
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in (manifest["checkpoint_name"], manifest["physical_guard_name"]):
            require(name in archive.namelist(), "Missing archive member: " + name)
        require(archive.getinfo(manifest["checkpoint_name"]).file_size
                == manifest["checkpoint_bytes"], "Unexpected checkpoint size")
        saved = archive.read(manifest["checkpoint_name"])
        require(len(saved) == manifest["checkpoint_bytes"], "Checkpoint size mismatch")
        require(hashlib.sha256(saved).hexdigest() == manifest["checkpoint_sha256"],
                "Checkpoint SHA256 mismatch")
        require(len(saved) > 8, "Safetensors header missing")
        length = struct.unpack("<Q", saved[:8])[0]
        require(0 < length < 1_000_000 and length + 8 <= len(saved),
                "Invalid Safetensors header")
        hdr = json.loads(saved[8:8+length])
        require(len([k for k in hdr if k != "__metadata__"]) == manifest["tensor_count"],
                "Safetensors tensor count mismatch")
        active = hdr.get(manifest["active_up_tensor"])
        require(active is not None, "Active FP16 tensor missing")
        require(active["dtype"] == manifest["active_up_dtype"], "Active dtype changed")
        require(active["shape"] == manifest["active_up_shape"], "Active shape changed")
        start, end = active["data_offsets"]
        require(0 <= start < end <= len(saved) - length - 8,
                "Active tensor offsets outside data")
        guard = json.loads(archive.read(manifest["physical_guard_name"]))
        require(guard["physical_FP16_sha256"] == manifest["checkpoint_sha256"],
                "Physical guard SHA256 mismatch")
        require(guard["physical_FP16_reload_pass"] is True, "Physical FP16 reload not verified")
        require(guard["real_original_456_and_S122_128_protected"] is True,
                "456+128 physical guard missing")
        require(guard["committed_gain_no_forgetting"] is True, "Historical skills lost")
        require(guard["real_S140_J7_transaction"]["committed"] is True,
                "J7 not physically committed")
        require(guard["old_456_full"] == manifest["old_full_after"],
                "Old 456 score mismatch")
        require(guard["S122_128_full"] == manifest["new_full_after"],
                "New 128 score mismatch")
        require(guard["CPU_speed_ratio"] >= manifest["min_cpu_ratio"], "CPU speed gate failed")
        require(guard["physical_bpp"] <= manifest["max_bpp"], "BPP gate failed")
        require(guard["model_accepted"] is False and guard["no_QA_no_LIVE"] is True,
                "LIVE/model promotion prohibited")
    return {
        "valid": True, "source_run": manifest["source_run_id"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "old_full": guard["old_456_full"], "new_full": guard["S122_128_full"],
        "bpp": guard["physical_bpp"], "cpu_ratio": guard["CPU_speed_ratio"],
        "physical_456_128_replay": "attested_by_original_S142_CI_only",
        "heldout_QA": "not_run", "model_accepted": False,
    }


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Do not forward REST API bearer token to redirected artifact storage host."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        out = super().redirect_request(req, fp, code, msg, headers, newurl)
        if out and (urllib.parse.urlsplit(newurl).hostname
                    != urllib.parse.urlsplit(req.full_url).hostname):
            out.remove_header("Authorization")
        return out


def fetch_artifact(manifest: dict, token: str) -> bytes:
    require(bool(token), "GitHub actions:read token required")
    url = ("https://api.github.com/repos/Kamara229/JessicaCore/actions/artifacts/"
           + str(manifest["source_artifact_id"]) + "/zip")
    req = urllib.request.Request(url, headers={
        "Authorization": "Bearer " + token,
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "calibri-s144-source-preflight",
    })
    with urllib.request.build_opener(SafeRedirect()).open(req, timeout=40) as response:
        data = response.read(5_000_001)
    require(len(data) <= 5_000_000, "Oversized artifact")
    return data


def main() -> None:
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
    blob = (Path(sys.argv[1]).read_bytes() if len(sys.argv) > 1
            else fetch_artifact(manifest, os.getenv("GITHUB_TOKEN", "")))
    print(json.dumps(verify_archive(blob, manifest), sort_keys=True))


if __name__ == "__main__":
    main()
