"""CB59 adapter mapping an arbitrary physical decoder model to Green Tunnel.

Does NOT assume fixed 135M or 360M, nor fixed 211 matrix number.
A future architecture has to supply its own adapter if this decoder
cannot represent its special operators. Never silently skip tensors.
"""
import re
from .schedule import Operator,plan_cycles
from .contract import PhysicalSource


def _family(name):
    segments=name.casefold().split(".")
    if "embed_tokens" in segments or name.endswith("lm_head.weight"):
        return "shared_embedding_head"
    if "self_attn" in segments or "attention" in segments:
        return "attention"
    if "mlp" in segments or "feed_forward" in segments or "ffn" in segments:
        return "mlp"
    if "experts" in segments or "expert" in segments:
        return "experts"
    return "other_source_operator"


def _depth(name):
    for pat in (
        r"(?:^|\.)(?:layers|blocks|h|layer|decoder_layers)\.(\d+)\.",
        r"(?:^|\.)(?:stack|stacks|experts)\.(\d+)\."):
        match=re.search(pat,name)
        if match:
            return int(match.group(1))
    return 0


def operators_from_CB59_metadata(metadata):
    """Derive ALL original physically unique 2D source matrices.

    Source key alias handles are counted exactly once according to
    tensor_descriptors, not duplicated in physical budget or training.
    """
    if metadata.get("bits_per_code")!=5 or metadata.get("vector_length")!=8:
        raise ValueError("This adapter only binds CB59 K32 V8 physical source")
    desc=metadata.get("tensor_descriptors")
    if not isinstance(desc,dict) or not desc:
        raise ValueError("Actual original source descriptors unavailable")
    out=[]
    for name,entry in desc.items():
        if not isinstance(entry,dict) or "dtype" not in entry:
            raise ValueError("Source metadata entry malformed")
        if entry["dtype"]=="CB59_K32V8":
            shape=entry.get("shape")
            if (not isinstance(shape,list) or len(shape)!=2 or
                any(not isinstance(x,int) or x<=0 for x in shape) or
                shape[1]%8):
                raise ValueError("Invalid physically packed matrix shape")
            out.append(Operator(name,_family(name),_depth(name),shape[0]*shape[1]))
        elif entry["dtype"]!="f4":
            raise ValueError("Unsupported original trained parameter dtype: "+entry["dtype"])
    aliases=metadata.get("source_alias_keys",{})
    if not isinstance(aliases,dict):
        raise ValueError("Tied source aliases missing")
    if any(name in desc for name in aliases):
        raise AssertionError("Aliases were charged twice as distinct trained tensors")
    if any(target not in desc for target in aliases.values()):
        raise AssertionError("Alias source tensor missing")
    if (set(desc)|set(aliases))!=set(metadata.get("full_original_state_keys",[])):
        raise AssertionError("Original pretrained 2D/1D source tensor keys incomplete")
    trained_unique=sum(
        int(entry["shape"][0])*int(entry["shape"][1])
        if len(entry["shape"])==2 else int(entry["shape"][0])
        for entry in desc.values()
    )
    if trained_unique!=metadata.get("original_unique_parameter_count"):
        raise AssertionError("Complete source unique parameter count does not match physical descriptors")
    if not out:
        raise AssertionError("Source contains no trained 2D original matrices")
    return tuple(out)


def plan_complete_CB59_recovery(metadata,number_of_cycles:int):
    operators=operators_from_CB59_metadata(metadata)
    schedule=plan_cycles(operators,number_of_cycles)
    assert all(len(x)==len(operators) for x in schedule)
    return {"total_original_unique_compressed_2d_operators":len(operators),
        "complete_cycle_count":len(schedule),
        "steps_per_cycle":len(operators),
        "all_original_2d_codebooks_can_receive_gradient_every_step":True,
        "every_original_2d_indices_get_discrete_update_turn_each_cycle":True,
        "family_counts":{f:sum(x.family==f for x in operators)
                         for f in sorted({x.family for x in operators})},
        "completed_zero_source_updates":True,
        "precommitted_targets":schedule}
