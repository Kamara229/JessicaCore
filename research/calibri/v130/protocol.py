"""Predeclared prompt-grounded reversal pairs; no factual memory required.

Swapping two associations preserves the word inventory and answer options.
Each half is scored, so a model that always says the same value cannot pass.
This is a *diagnostic* set, NOT a substitute for a frozen blind TEST split.
"""
MAPPINGS = (
    ("mivi-saro", "Mivi", "Saro", "amber", "cobalt"),
    ("dela-fenn", "Dela", "Fenn", "copper", "silver"),
    ("pavo-kori", "Pavo", "Kori", "north", "south"),
    ("reni-toma", "Reni", "Toma", "circle", "square"),
    ("bilo-wena", "Bilo", "Wena", "tiger", "eagle"),
    ("naro-zemi", "Naro", "Zemi", "violet", "orange"),
)


def cases():
    rows = []
    for pair, left, right, value_a, value_b in MAPPINGS:
        for orientation in ("original", "swapped"):
            lv, rv = ((value_a, value_b) if orientation == "original"
                      else (value_b, value_a))
            ledger = (f"Use this invented ledger only, not outside knowledge: "
                      f"{left} means {lv}; {right} means {rv}. ")
            for focus, name, answer, alternative in (
                ("left", left, lv, rv), ("right", right, rv, lv)
            ):
                question = (ledger + f"What is the value of {name}? "
                            "Reply with exactly one lowercase word.")
                rows.append({"id": f"{pair}:{focus}:{orientation}",
                             "pair": pair, "focus": focus,
                             "orientation": orientation, "question": question,
                             "gold": answer, "foil": alternative})
    if len(rows) != 4 * len(MAPPINGS):
        raise AssertionError("not a complete paired swap experiment")
    if len({r["id"] for r in rows}) != len(rows):
        raise AssertionError("duplicate paired case")
    return tuple(rows)


def shuffled_questions(rows):
    """Counterfactual controller condition; no correct label is ever passed."""
    # A fixed derangement that puts the controller on a DIFFERENT record.
    rotate = 7
    n = len(rows)
    if n <= rotate:
        raise ValueError("not enough independent records")
    return {r["id"]: rows[(i + rotate) % n]["question"]
            for i, r in enumerate(rows)}
