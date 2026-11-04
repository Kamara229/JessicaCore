"""Fixed before fitting. Free generation never receives a reference answer."""
from v117.protocol import roles

TRAIN = (
    ("Give only the difference of 43 and 16.", "27"),
    ("Compute 65 minus 38. Give the number.", "27"),
    ("What is 15 minus 19? Give only the number.", "-4"),
    ("Multiply 11 by 5. Answer with the number.", "55"),
    ("Add 41 and 17. Answer with the number.", "58"),
    ("Divide 63 by 7. Answer with the number.", "9"),
    ("Name the capital city of Peru.", "Lima"), ("Name the capital city of Poland.", "Warsaw"),
    ("Name the capital city of Cuba.", "Havana"), ("Name the capital city of Iceland.", "Reykjavik"),
    ("Name the capital city of Thailand.", "Bangkok"), ("Name the capital city of Ukraine.", "Kyiv"),
    ("Give the chemical symbol for helium.", "He"), ("Give the chemical symbol for neon.", "Ne"),
    ("Give the chemical symbol for zinc.", "Zn"), ("Which star does Earth orbit? Give its name.", "Sun"),
    ("What is frozen water called? Give one word.", "Ice"), ("What color is a ripe lemon? Give one word.", "Yellow"),
    ("Is a dozen greater than ten? Reply Yes or No.", "Yes"), ("Is ice a form of water? Reply Yes or No.", "Yes"),
    ("Is a robin a bird? Reply Yes or No.", "Yes"), ("Is a cube flat? Reply Yes or No.", "No"),
    ("Does a year contain ten months? Reply Yes or No.", "No"), ("Is nine an even number? Reply Yes or No.", "No"),
)
VALID = (
    ("Subtract 29 from 53. Give only the number.", "24"), ("What is 13 minus 20? Give only the number.", "-7"),
    ("Multiply 8 by 11. Give the number.", "88"), ("Add 35 and 26. Give the number.", "61"),
    ("Name the capital city of Nepal.", "Kathmandu"), ("Name the capital city of Ghana.", "Accra"),
    ("Name the capital city of Vietnam.", "Hanoi"), ("Name the capital city of Senegal.", "Dakar"),
    ("Give the chemical symbol for magnesium.", "Mg"), ("Give the chemical symbol for nickel.", "Ni"),
    ("What is the name of Earth's natural satellite? Give one word.", "Moon"),
    ("What color is fresh snow? Give one word.", "White"),
    ("Is a salmon a fish? Reply Yes or No.", "Yes"), ("Is eleven a prime number? Reply Yes or No.", "Yes"),
    ("Is Saturday followed by Friday? Reply Yes or No.", "No"), ("Is four an odd number? Reply Yes or No.", "No"),
)
OPEN = ("In one sentence, explain why a wet towel dries on a clothesline.",
        "Give one concrete way to keep a library book dry in the rain.")
SEED = 129
LR = 0.001


def layout(moved):
    if not moved:
        return [(n, r, c, 1) for n, r, c in roles()]
    entries = []
    for layer in range(30):
        attn, rows, rank = (("q", 576, 1), ("k", 192, 2), ("v", 192, 1))[layer % 3]
        mlp = ("gate", "up")[layer % 2]
        prefix = f"model.layers.{layer}."
        entries.extend(((prefix + f"self_attn.{attn}_proj", rows, 576, rank),
                        (prefix + f"mlp.{mlp}_proj", 1536, 576, 1)))
    return entries
