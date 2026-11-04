"""Test PROBES are previously seen lab diagnostics; no independent TEST."""
from v119.protocol import VALID,OPEN
TASKS=(VALID[0],VALID[2],VALID[3],VALID[4],VALID[6],VALID[7],VALID[10],VALID[11])
OPEN_TASKS=(OPEN[0],OPEN[2])
SCHEDULE=(
    ("original_paid_joint",(1.0,1.0)),
    ("disable_all_corrective_roles",(0.0,0.0)),
    ("boost_all_corrective_roles",(2.0,2.0)),
    ("favor_mlp_over_attention",(0.5,1.5)),
)
assert len(TASKS)==8 and len(OPEN_TASKS)==2 and len(SCHEDULE)==4
