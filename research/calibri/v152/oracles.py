"""Per-layer FP32 swap: diagnostic only, NOT a compressed candidate."""
from contextlib import contextmanager

@contextmanager
def replace_one_layer(model,teacher,index):
    if not 0<=index<30:
        raise ValueError("invalid frozen original transformer index")
    if len(model.model.layers)!=30 or len(teacher.model.layers)!=30:
        raise ValueError("model architecture changed")
    original=model.model.layers[index]
    borrowed=teacher.model.layers[index]
    if borrowed is original:
        raise AssertionError("expected independent original pretrained module")
    bytes_borrowed=sum(p.numel()*p.element_size()
                       for p in borrowed.parameters())
    if bytes_borrowed<100000:
        raise ValueError("teacher oracle block is not truly pretrained")
    try:
        model.model.layers[index]=borrowed
        yield {"FP32_borrowed_bytes":bytes_borrowed,
               "physical_bpp_requirement_is_VIOLATED":True,
               "not_a_native_packed_candidate":True,
               "not_a_speed_benchmark":True}
    finally:
        model.model.layers[index]=original
        if model.model.layers[index] is not original:
            raise AssertionError("borrowed layer was not properly restored")
