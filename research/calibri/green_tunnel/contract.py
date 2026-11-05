"""Codec/model-independent physical source invariant (no training dependency).

Green Tunnel may optimize ONLY bits that the exported physical archive
actually contains. No source can claim <=0.7bpp based on tensor RAM,
fake byte estimates, omitted aliases, hidden teachers or sidecars.
"""
from dataclasses import dataclass
import hashlib
from typing import Callable, Any, Mapping

@dataclass(frozen=True)
class PhysicalSource:
    codec: str
    original_unique_parameters: int
    physical_whole_bytes: int
    physical_sha256: str
    maximum_bpp: float = 0.700
    aliases: tuple[tuple[str,str], ...] = ()
    def __post_init__(self):
        if not self.codec or self.original_unique_parameters<=0 or self.physical_whole_bytes<=0:
            raise ValueError("A complete physical trained source is required")
        if len(self.physical_sha256)!=64 or any(c not in "0123456789abcdef" for c in self.physical_sha256):
            raise ValueError("Whole-source SHA256 must be a lowercase 64-digit digest")
        if not (0<self.maximum_bpp<=0.700):
            raise ValueError("0.700 source cap is the absolute requested target")
        names=[x for x,_ in self.aliases]
        if len(names)!=len(set(names)):
            raise ValueError("Duplicate tied-source alias")
        if any(a==b or not a or not b for a,b in self.aliases):
            raise ValueError("Invalid trained parameter alias")
    @property
    def actual_bpp(self)->float:
        return (8*self.physical_whole_bytes)/self.original_unique_parameters
    @property
    def physical_byte_cap(self)->int:
        return (self.original_unique_parameters*7)//80
    @property
    def valid_rate(self)->bool:
        return self.physical_whole_bytes<=self.physical_byte_cap


def verify_physical_source(
    source:PhysicalSource,
    complete_binary:bytes,
    decode:Callable[[bytes],Any]|None=None,
    verify_alias:Callable[[Any,tuple[tuple[str,str],...]],bool]|None=None
)->dict:
    """Exact physically paid archive check before any source quality claim."""
    if not isinstance(complete_binary,bytes):
        raise TypeError("Use FULL authenticated physical binary bytes, not a view or estimate")
    digest=hashlib.sha256(complete_binary).hexdigest()
    if len(complete_binary)!=source.physical_whole_bytes or digest!=source.physical_sha256:
        raise ValueError("Physical source bytes/SHA drift")
    if not source.valid_rate:
        raise AssertionError("Source exceeds complete <=0.700 bpp threshold")
    decoded=decode(complete_binary) if decode is not None else None
    if verify_alias is not None:
        if decode is None or not verify_alias(decoded,source.aliases):
            raise AssertionError("Lost shared trained source alias in decoder")
    return {"physical_size_bytes":len(complete_binary),
        "original_unique_parameters":source.original_unique_parameters,
        "actual_complete_bpp":source.actual_bpp,
        "complete_source_sha256":digest,
        "physically_paid":True,
        "codec":source.codec,
        "aliases_verified":bool(verify_alias is not None)}


def require_paid_update(before:PhysicalSource,after:PhysicalSource,
                        actual_written_binary:bytes)->dict:
    """Allow learned-index/centroid rewrites, never an uncharged neural tunnel."""
    if before.codec!=after.codec or before.original_unique_parameters!=after.original_unique_parameters:
        raise AssertionError("Source/architecture switch inside recovery loop")
    if before.aliases!=after.aliases:
        raise AssertionError("Tied original trained parameter aliases changed")
    audited=verify_physical_source(after,actual_written_binary)
    if not before.valid_rate or not after.valid_rate:
        raise AssertionError("Unpaid recovery capacity added")
    return {**audited,"pretraining_original_source_sha256":before.physical_sha256,
        "source_bits_per_original_parameter_before":before.actual_bpp,
        "source_bits_per_original_parameter_after":after.actual_bpp,
        "new_green_tunnel_inference_weights_outside_binary":0,
        "same_complete_original_unique_parameters":True}
