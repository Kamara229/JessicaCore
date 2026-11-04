"""New synthetic same-prefix diagnostics, not official TEST nor candidate fitting."""
PROBES=(
 ("arithmetic","Question: What is eleven plus four? Answer:"),
 ("factual","Question: What is the capital of Portugal? Answer:"),
)
ROW_CHUNK=256
EXPLICIT_ROW_INDICES=(0,1,7,8,31,32,255,256,1023,1024,4095,4096,49151)
MAX_NATIVE_VS_PYTHON_ABS_DIFF=0.003
assert len({p for _,p in PROBES})==len(PROBES)
