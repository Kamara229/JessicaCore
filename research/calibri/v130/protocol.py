"""New counterfactual paired prompts: same word multiset, different correct fact.

DIAGNOSTIC ONLY: these tasks are not the independent locked TEST.
No gold answer is supplied to generation or to the frozen Green policy.
"""
from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Pair:
    name: str
    family: str
    left: str
    left_answer: str
    right: str
    right_answer: str


PAIRS = (
    Pair("key_color", "binding",
         "On the fictional desk, Mira has the blue key and Niko has the red key. What color is Mira's key? Reply with one word.",
         "blue",
         "On the fictional desk, Mira has the red key and Niko has the blue key. What color is Mira's key? Reply with one word.",
         "red"),
    Pair("animal_owner", "binding",
         "In the story, Arin owns a cat and Bela owns a dog. Which animal does Arin own? Reply with one word.",
         "cat",
         "In the story, Arin owns a dog and Bela owns a cat. Which animal does Arin own? Reply with one word.",
         "dog"),
    Pair("train_station", "binding",
         "In this fictional timetable, train A stops at Lotus and train B stops at Cedar. Where does train A stop? Reply with one word.",
         "Lotus",
         "In this fictional timetable, train A stops at Cedar and train B stops at Lotus. Where does train A stop? Reply with one word.",
         "Cedar"),
    Pair("code_number", "binding",
         "In this invented codebook, code X means 19 and code Y means 52. What does code X mean? Reply with one number.",
         "19",
         "In this invented codebook, code X means 52 and code Y means 19. What does code X mean? Reply with one number.",
         "52"),
    Pair("race_order", "relation",
         "In the fictional race, Asha finished before Eren. Who finished first? Reply with one name.",
         "Asha",
         "In the fictional race, Eren finished before Asha. Who finished first? Reply with one name.",
         "Eren"),
    Pair("bag_yesno", "yes_no",
         "In the story, Mia carries a green bag and Tim carries a yellow bag. Does Mia carry a green bag? Reply Yes or No.",
         "Yes",
         "In the story, Mia carries a yellow bag and Tim carries a green bag. Does Mia carry a green bag? Reply Yes or No.",
         "No"),
    Pair("fruit_owner", "binding",
         "At the fictional market, Lio bought apples and Tia bought pears. What did Lio buy? Reply with one word.",
         "apples",
         "At the fictional market, Lio bought pears and Tia bought apples. What did Lio buy? Reply with one word.",
         "pears"),
    Pair("box_contents", "binding",
         "On the fictional shelf, box A contains salt and box B contains rice. What is in box A? Reply with one word.",
         "salt",
         "On the fictional shelf, box A contains rice and box B contains salt. What is in box A? Reply with one word.",
         "rice"),
)


def words(question):
    return sorted(re.findall(r"[a-z0-9]+", question.casefold()))


def validate():
    assert len({p.name for p in PAIRS}) == len(PAIRS)
    for p in PAIRS:
        assert p.left_answer.casefold() != p.right_answer.casefold()
        assert words(p.left) == words(p.right), p.name
        assert p.left != p.right
        assert p.left_answer.casefold() in words(p.left)
        assert p.right_answer.casefold() in words(p.right)


def tasks():
    validate()
    for pair in PAIRS:
        yield (pair, "left", pair.left, pair.left_answer, pair.right_answer, pair.right)
        yield (pair, "right", pair.right, pair.right_answer, pair.left_answer, pair.left)


validate()
