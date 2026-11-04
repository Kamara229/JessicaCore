"""CALIBRI v136. Read-only FP32 oracle swap to isolate early operator failure.

Oracle arms borrow many FP32 matrices, intentionally violating compactness.
Only a *fault diagnosis*, never a 0.7 bpp deployable model or speed comparison.
"""
from contextlib import contextmanager, ExitStack

ARMS = (
    "CB59",
    "oracle_input",
    "oracle_block0",
    "oracle_input_block0",
    "oracle_input_attn0",
    "oracle_input_mlp0",
    "oracle_input_first5",
    "oracle_input_first5_head",
)

def parameters(arm):
    if arm not in ARMS:
        raise ValueError("unregistered oracle condition")
    return {
        "embedding": arm.startswith("oracle_input"),
        "head": arm.endswith("_head"),
        "first5": "first5" in arm,
        "block0": arm.endswith("_block0"),
        "attn0": arm.endswith("_attn0"),
        "mlp0": arm.endswith("_mlp0"),
    }


@contextmanager
def teacher_components(student, teacher, arm):
    """Replace original packed modules only inside this lexical context."""
    flags = parameters(arm)
    if len(student.model.layers)!=30 or len(teacher.model.layers)!=30:
        raise ValueError("expected matched 30-layer full models")
    with ExitStack() as restore:
        def swap(owner, name, replacement):
            original=getattr(owner,name)
            restore.callback(setattr,owner,name,original)
            setattr(owner,name,replacement)
        if flags["embedding"]:
            swap(student.model,"embed_tokens",teacher.model.embed_tokens)
        if flags["head"]:
            swap(student,"lm_head",teacher.lm_head)
        if flags["first5"]:
            for depth in range(5):
                # ModuleList implements indexed write with automatic module tracking.
                old=student.model.layers[depth]
                restore.callback(student.model.layers.__setitem__,depth,old)
                student.model.layers[depth]=teacher.model.layers[depth]
        if flags["block0"]:
            old=student.model.layers[0]
            restore.callback(student.model.layers.__setitem__,0,old)
            student.model.layers[0]=teacher.model.layers[0]
        if flags["attn0"]:
            swap(student.model.layers[0],"self_attn",teacher.model.layers[0].self_attn)
        if flags["mlp0"]:
            swap(student.model.layers[0],"mlp",teacher.model.layers[0].mlp)
        yield flags


def expected_physical_oracle_bytes(teacher, arm):
    """Estimate FP32 teacher weights borrowed; excludes native source/overlap."""
    flags=parameters(arm)
    names=[]
    if flags["embedding"]: names.append(("input",teacher.model.embed_tokens))
    if flags["head"]: names.append(("output",teacher.lm_head))
    if flags["first5"]:
        names.extend((f"layer{i}",teacher.model.layers[i]) for i in range(5))
    elif flags["block0"]:
        names.append(("layer0",teacher.model.layers[0]))
    elif flags["attn0"]:
        names.append(("attn0",teacher.model.layers[0].self_attn))
    elif flags["mlp0"]:
        names.append(("mlp0",teacher.model.layers[0].mlp))
    # Count unique object IDs for tied input/output parameters.
    ids={}
    for prefix,module in names:
        for _,p in module.named_parameters():
            ids[id(p)]=p
    return sum(p.numel()*p.element_size() for p in ids.values())
