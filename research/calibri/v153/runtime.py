"""Direct CPU K32xK2 extra onebit residual for original trained linear ops.

Reuse the already validated v141 grouped LUT C ABI, but make it available
for *all* 7 original linear layers of ONE selected transformer block, not
only the 49152x576 tied vocab. Original full 2D weights remain packed.
"""
from contextlib import contextmanager
import ctypes
from pathlib import Path
import numpy as np
from direct_runtime_v59 import PackedLinear
from v141.kernel import compile_lut
from v140.runtime import OutputHead,PTR,SIZE,I64


class GeneralNativeLUT:
    def __init__(self,library,handle,patch):
        self.lib=library
        self.handle=handle
        self.base=np.ascontiguousarray(handle.packed,dtype=np.uint8)
        self.bits=np.ascontiguousarray(patch.bits,dtype=np.uint8)
        self.centers=np.ascontiguousarray(patch.centers,dtype=np.float32)
        self.scales=np.ascontiguousarray(handle.row_scales,dtype=np.float32)
        self.rows,self.dim=handle.shape
        if (self.dim%8 or self.rows<1 or self.dim<8 or
                self.centers.shape!=(32,2,8) or
                len(self.bits)!=(self.rows*(self.dim//8)+7)//8):
            raise ValueError("original matrix not represented by paid K32+1bit")
        self.fn=library.calibri141_output_delta
        self.fn.argtypes=[
            PTR,SIZE,PTR,SIZE,PTR,PTR,PTR,I64,I64,I64,PTR]
        self.fn.restype=ctypes.c_int

    @staticmethod
    def p(arr):
        return PTR(int(arr.ctypes.data))

    def linear(self,inputs):
        x=np.ascontiguousarray(inputs,dtype=np.float32).reshape(-1,self.dim)
        y=np.empty((len(x),self.rows),dtype=np.float32)
        code=self.fn(self.p(self.base),len(self.base),
                     self.p(self.bits),len(self.bits),
                     self.p(self.centers),self.p(self.scales),
                     self.p(x),len(x),self.rows,self.dim,self.p(y))
        if code or not np.isfinite(y).all():
            raise RuntimeError("invalid original packed paid 1bit layer GEMV: "+str(code))
        return y


@contextmanager
def install_paid_block(model,archive,folder):
    """Never mutate original CB59 archive or pretrained weights in memory."""
    filename=Path(folder)/"calibri153_original_layer_residual.so"
    library=ctypes.CDLL(str(compile_lut(filename)))
    replaced=[]
    try:
        for name,patch in sorted(archive.patches.items()):
            module_path=name[:-len(".weight")]
            parent_path,_,field=module_path.rpartition(".")
            owner=model.get_submodule(parent_path)
            original=getattr(owner,field)
            if not isinstance(original,PackedLinear):
                raise ValueError("required original native packed operator absent: "+name)
            if original.handle.shape!=patch.shape:
                raise ValueError("source codeword geometry changed")
            runner=GeneralNativeLUT(library,original.handle,patch)
            replacement=OutputHead(original,runner)
            setattr(owner,field,replacement)
            replaced.append((owner,field,original))
        if len(replaced)!=len(archive.patches):
            raise ValueError("not all original 2D operators refined")
        if any(p.ndim>=2 for p in model.parameters()):
            raise ValueError("dense original pretrained matrices persisted")
        yield {"paid_transformed_original_block":archive.layer,
               "source_only_no_dense_original":True,
               "paid_2D_modules":len(replaced)}
    finally:
        for owner,field,original in reversed(replaced):
            setattr(owner,field,original)
