"""All original bitplanes remain PAID, despite RAM-only operator toggles."""
from dataclasses import replace
from .protocol import MATRIX_KEYS,SHORT,ARMS,check


def all_predeclared_arms(physical):
    check()
    if set(physical.patches)!=set(MATRIX_KEYS) or physical.layer!=0:
        raise ValueError("not real original fully paid first-block source")
    operators=dict(zip(SHORT,MATRIX_KEYS))
    result={ARMS[0]:(),ARMS[1]:tuple(MATRIX_KEYS)}
    for short,path in operators.items():
        result[f"paid_ONLY_{short}"]=(path,)
    for short,path in operators.items():
        result[f"paid_EXCEPT_{short}"]=tuple(
            x for x in MATRIX_KEYS if x!=path)
    if tuple(result)!=ARMS:
        raise ValueError("diagnostic arms reordered, dropped or postselected")
    return {arm:replace(physical,patches={key:physical.patches[key] for key in enabled})\n            for arm,enabled in result.items()}
