"""Gold-free Russian UD grammar bridge tests: no model, corpus or EVAL."""
import unittest

from research.nra2_bridge.stage3ck_russian_ud_role_bridge import (
    UDWord, EntityMention, GrammarContractError, extract_grammar,
)


def annotated(text, specs):
    """Build test annotations from literal text, never training labels."""
    words = []
    pos = 0
    for i, (surface, lemma, upos, head, dep, *extra) in enumerate(specs, 1):
        start = text.index(surface, pos)
        pos = start + len(surface)
        words.append(UDWord(i, surface, lemma, upos, head, dep,
                            start, pos, tuple(extra[0]) if extra else ()))
    return tuple(words)


class RussianGrammarContractTests(unittest.TestCase):
    def active(self, text, subject_first):
        # The surface text and true UD dependency arcs differ in word order.
        if subject_first:
            w = annotated(text, [
                ("Иван", "Иван", "PROPN", 2, "nsubj"),
                ("увидел", "увидеть", "VERB", 0, "root"),
                ("Марию", "Мария", "PROPN", 2, "obj"),
                (".", ".", "PUNCT", 2, "punct"),
            ])
            mentions = (EntityMention("R0", 1), EntityMention("R1", 3))
        else:
            w = annotated(text, [
                ("Марию", "Мария", "PROPN", 2, "obj"),
                ("увидел", "увидеть", "VERB", 0, "root"),
                ("Иван", "Иван", "PROPN", 2, "nsubj"),
                (".", ".", "PUNCT", 2, "punct"),
            ])
            mentions = (EntityMention("R1", 1), EntityMention("R0", 3))
        return extract_grammar(text, w, mentions)

    def test_word_order_invariance(self):
        a = self.active("Иван увидел Марию.", True)
        b = self.active("Марию увидел Иван.", False)
        self.assertEqual(a.role_entities("subject"), ("R0",))
        self.assertEqual(b.role_entities("subject"), ("R0",))
        self.assertEqual(a.role_entities("direct_object"), ("R1",))
        self.assertEqual(b.role_entities("direct_object"), ("R1",))
        self.assertIsNone(a.semantic_direction)
        self.assertIsNone(a.graph_reachability)

    def test_real_reversal_is_not_a_paraphrase(self):
        text = "Мария увидела Ивана."
        w = annotated(text, [
            ("Мария", "Мария", "PROPN", 2, "nsubj"),
            ("увидела", "увидеть", "VERB", 0, "root"),
            ("Ивана", "Иван", "PROPN", 2, "obj"),
            (".", ".", "PUNCT", 2, "punct"),
        ])
        a = extract_grammar(text, w, (EntityMention("R1", 1),
                                      EntityMention("R0", 3)))
        self.assertEqual(a.role_entities("subject"), ("R1",))
        self.assertEqual(a.role_entities("direct_object"), ("R0",))

    def test_passive_voice_not_reversed_by_surface_order(self):
        text = "Иван был замечен Марией."
        w = annotated(text, [
            ("Иван", "Иван", "PROPN", 3, "nsubj:pass"),
            ("был", "быть", "AUX", 3, "aux"),
            ("замечен", "заметить", "VERB", 0, "root", ("Voice=Pass",)),
            ("Марией", "Мария", "PROPN", 3, "obl:agent"),
            (".", ".", "PUNCT", 3, "punct"),
        ])
        f = extract_grammar(text, w, (EntityMention("R0", 1),
                                      EntityMention("R1", 4)))
        self.assertEqual(f.role_entities("passive_subject"), ("R0",))
        self.assertEqual(f.role_entities("oblique_agent"), ("R1",))
        self.assertEqual(len(f.arcs), 4)

    def test_pronouns_unresolved_without_coreference(self):
        text = "Она увидела его."
        w = annotated(text, [
            ("Она", "она", "PRON", 2, "nsubj", ("Case=Nom",)),
            ("увидела", "увидеть", "VERB", 0, "root"),
            ("его", "он", "PRON", 2, "obj", ("Case=Acc",)),
            (".", ".", "PUNCT", 2, "punct"),
        ])
        f = extract_grammar(text, w)
        self.assertEqual(f.unresolved_pronouns, (1, 3))
        self.assertEqual(f.role_entities("subject"), ())
        self.assertEqual(f.role_entities("direct_object"), ())
        with self.assertRaises(GrammarContractError):
            extract_grammar(text, w, (EntityMention("R1", 1),))

    def test_adjective_noun_link(self):
        text = "Высокий Иван увидел Марию."
        w = annotated(text, [
            ("Высокий", "высокий", "ADJ", 2, "amod", ("Case=Nom",)),
            ("Иван", "Иван", "PROPN", 3, "nsubj"),
            ("увидел", "увидеть", "VERB", 0, "root"),
            ("Марию", "Мария", "PROPN", 3, "obj"),
            (".", ".", "PUNCT", 3, "punct"),
        ])
        f = extract_grammar(text, w, (EntityMention("R0", 2),
                                      EntityMention("R1", 4)))
        self.assertIn((2, 1, "amod"),
                      [(x.head_id, x.child_id, x.relation) for x in f.arcs])
        self.assertEqual(f.words[0].upos, "ADJ")

    def test_grammatical_subject_is_not_semantic_source(self):
        # In "R1 received from R0" the subject is the recipient. Returning
        # subject as graph source would reverse a directed relation.
        text = "R1 получил письмо от R0."
        w = annotated(text, [
            ("R1", "R1", "PROPN", 2, "nsubj"),
            ("получил", "получить", "VERB", 0, "root"),
            ("письмо", "письмо", "NOUN", 2, "obj"),
            ("от", "от", "ADP", 5, "case"),
            ("R0", "R0", "PROPN", 2, "obl"),
            (".", ".", "PUNCT", 2, "punct"),
        ])
        f = extract_grammar(text, w, (EntityMention("R1", 1),
                                      EntityMention("R0", 5)))
        self.assertEqual(f.role_entities("subject"), ("R1",))
        self.assertIsNone(f.semantic_direction)
        self.assertIsNone(f.graph_reachability)

    def test_reject_malformed_or_untrusted_annotations(self):
        text = "Иван увидел Марию."
        f = self.active(text, True)
        with self.assertRaises(GrammarContractError):
            extract_grammar(text, f.words, language="en")
        with self.assertRaises(GrammarContractError):
            extract_grammar(text, f.words, (EntityMention("R999", 1),))
        broken = list(f.words)
        broken[0] = UDWord(1, "Иван", "Иван", "PROPN", 1, "nsubj",
                           0, 4)
        with self.assertRaises(GrammarContractError):
            extract_grammar(text, tuple(broken))
        broken = list(f.words)
        broken[2] = UDWord(3, "Мария", "Мария", "PROPN", 2, "obj",
                           11, 17)
        with self.assertRaises(GrammarContractError):
            extract_grammar(text, tuple(broken))

    def test_multiple_verbs_abstain(self):
        text = "Иван сказал что Мария ушла."
        w = annotated(text, [
            ("Иван", "Иван", "PROPN", 2, "nsubj"),
            ("сказал", "сказать", "VERB", 0, "root"),
            ("что", "что", "SCONJ", 5, "mark"),
            ("Мария", "Мария", "PROPN", 5, "nsubj"),
            ("ушла", "уйти", "VERB", 2, "ccomp"),
            (".", ".", "PUNCT", 2, "punct"),
        ])
        with self.assertRaises(GrammarContractError):
            extract_grammar(text, w)


if __name__ == "__main__":
    unittest.main()
