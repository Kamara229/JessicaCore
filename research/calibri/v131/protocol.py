"""Small preregistered structural subset of v130: no training and no locked TEST."""
from v130.protocol import PAIRS

CHOSEN = ("key_color", "animal_owner", "train_station", "code_number",
          "race_order", "bag_yesno", "fruit_owner", "box_contents")
MODES = ("CB59", "head_only", "body_only", "v119_neutral",
         "v119_green_current", "v119_green_swapped", "v119_green_constant")


def examples():
    selected = [p for p in PAIRS if p.name in CHOSEN]
    if len(selected) != len(CHOSEN):
        raise ValueError("missing counterfactual task")
    for p in selected:
        yield (p.name, "left", p.left, p.right, p.left_answer, p.right_answer)
        yield (p.name, "right", p.right, p.left, p.right_answer, p.left_answer)
