"""S96 gold-blind original→licensed UD native BPE TRAIN-only contract tests.

Pure constructed fake original offsets; NOT real original Calibri
grammar quality or real human-labelled events. No live/Jessica.
"""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import ast
import unittest

import torch

from research.calibri_nra3.s94_ud_taiga_trainonly_grammatical_source_gold_WALL import (
    sha256_serialized,
)
from research.calibri_nra3.s91_neural_multilayer_event_graph import (
    ROLE_NAMES,STATE_NAMES,
)
from research.calibri_nra3.s93_offline_ud_grammatical_role_train_loss_ONLY import (
    IGNORE,UNKNOWN,
)
from research.calibri_nra3 import s96_taiga_originalBPE_offline_UD_targets as s96


def make_synthetic_256(n=4):
    source=[{"row_id":f"S94_TAIGA_TRAIN_{i+1:03d}",
             "text":f"Курьер передал письмо {i}"}
            for i in range(256)]
    word_spans=[[0,6],[7,14],[15,21]]
    labels=[]
    for i,s in enumerate(source):
        roles=[-1]*8
        roles[0]=0;roles[1]=2
        labels.append({
            "row_id":s["row_id"],
            "predicate_UD_word_zero_index_TRAIN_ONLY":1,
            "role_argument_UD_word_zero_indices_TRAIN_ONLY":roles,
            "word_surface_start_end_char_offsets_TRAIN_ONLY":[list(z) for z in word_spans],
            "UD_voice_passive_TRAIN_ONLY":64<=i<128,
            "UD_negation_particle_link_present_TRAIN_ONLY":128<=i<192,
            "not_yet_original_native_BPE_coordinates":True,
            "reported_speech_source_and_coreference_GOLD_UNKNOWN":True,
            "not_independently_adjudicated_for_event_F1":True,
        })
    indexes=torch.tensor([0,63,64,128],dtype=torch.long)
    starts=torch.tensor([[0,7,15,-1]]*n,dtype=torch.int32)
    ends=torch.tensor([[6,14,21,-1]]*n,dtype=torch.int32)
    last=torch.tensor([[1,2,3,-1]]*n,dtype=torch.int32)
    safe=torch.tensor([[True,True,True,False]]*n,dtype=torch.bool)
    valid=torch.tensor([[True]*5+[False]]*n,dtype=torch.bool)
    return source,labels,indexes,starts,ends,last,safe,valid


def build(*,modify=None):
    source,labels,indices,starts,ends,last,safe,valid=make_synthetic_256()
    if modify: modify(source,labels,indices,starts,ends,last,safe,valid)
    with patch.object(s96,"S94_SOURCE_SHA",sha256_serialized(source)):
        return s96.map_UD_word_labels_ONLY_AFTER_source_features(
            source_rows=source,ud_labels=labels,
            source_valid=valid,source_row_ids=indices,
            raw_char_starts=starts,raw_char_ends=ends,
            raw_last_original_BPE=last,raw_causal_safe=safe,
            ud_gold_sha_expected=sha256_serialized(labels),
        )


class TestS96SourceThenUDAlignment(unittest.TestCase):
    def test_no_oracle_input_and_exact_real_BPE_event_argument_role(self):
        x=build()
        self.assertEqual(x.original_source_indices.tolist(),[0,63,64,128])
        self.assertEqual(x.eligible.tolist(),[True]*4)
        self.assertEqual(x.train.tolist(),[True,False,True,True])
        self.assertEqual(x.same_corpus_development.tolist(),[False,True,False,False])
        self.assertEqual(x.n_mapped_roles,8)
        role=x.targets.role_argument_token
        self.assertEqual(int(role[0,ROLE_NAMES.index("actor"),2]),1)
        self.assertEqual(int(role[0,ROLE_NAMES.index("patient"),2]),3)
        self.assertEqual(int(x.targets.mention[0,2]),1)
        self.assertEqual(int(x.targets.mention[0,1]),IGNORE)
        self.assertEqual(float(x.targets.state[0,2,STATE_NAMES.index("assertion")]),
                         float(UNKNOWN))

    def test_old_eight_neural_miniareas_keep_passive_negation_separate(self):
        x=build()
        self.assertEqual(float(x.targets.state[2,2,STATE_NAMES.index("passive")]),1.)
        self.assertEqual(float(x.targets.mini_area[2,2,2]),1.)
        self.assertEqual(float(x.targets.state[3,2,STATE_NAMES.index("negated")]),1.)
        self.assertEqual(float(x.targets.mini_area[3,2,1]),1.)
        self.assertEqual(float(x.targets.mini_area[0,2,7]),float(UNKNOWN))
        self.assertEqual(float(x.targets.state[0,2,STATE_NAMES.index("reported")]),float(UNKNOWN))
        self.assertTrue((x.targets.role_argument_token[0,3:]==IGNORE).all())

    def test_source_original_BPE_unsafe_event_fails_closed_no_leaked_ghost(self):
        def spoil(*args):
            safe=args[6]
            safe[2,1]=False
        x=build(modify=spoil)
        self.assertEqual(x.eligible.tolist(),[True,True,False,True])
        self.assertEqual(len(x.row_exclusions),1)
        self.assertEqual(x.row_exclusions[0]["original_source_index"],64)
        self.assertEqual(int(x.targets.mention[2,2]),IGNORE)
        self.assertEqual(int(x.targets.role_argument_token[2,0,2]),IGNORE)

    def test_wrong_UD_source_word_coordinates_rejected_not_guessed(self):
        def spoil(*args):
            labels=args[1]
            labels[63]["word_surface_start_end_char_offsets_TRAIN_ONLY"][1]=[7,13]
        x=build(modify=spoil)
        self.assertFalse(bool(x.eligible[1]))
        self.assertTrue(bool(x.eligible[0]))
        self.assertEqual(len(x.row_exclusions),1)

    def test_bad_annotation_sha_fails_before_any_label_mapping(self):
        src,gold,idx,starts,ends,last,safe,valid=make_synthetic_256()
        with patch.object(s96,"S94_SOURCE_SHA",sha256_serialized(src)):
            with self.assertRaisesRegex(ValueError,"SHA"):
                s96.map_UD_word_labels_ONLY_AFTER_source_features(
                    source_rows=src,ud_labels=gold,
                    source_valid=valid,source_row_ids=idx,
                    raw_char_starts=starts,raw_char_ends=ends,
                    raw_last_original_BPE=last,raw_causal_safe=safe,
                    ud_gold_sha_expected="a"*64,
                )

    def test_future_one_time_held_sets_and_runtime_parser_not_imported(self):
        f=Path("research/calibri_nra3/s96_taiga_originalBPE_offline_UD_targets.py")
        t=ast.parse(f.read_text())
        imports=[z.module or "" for z in ast.walk(t)
                 if isinstance(z,ast.ImportFrom)]
        self.assertFalse(any("s79" in x.lower() or "s86" in x.lower()
                             or "stanza" in x.lower() or "udpipe" in x.lower()
                             for x in imports))
        p=Path("research/calibri_nra3/s91_neural_multilayer_event_graph.py")
        self.assertNotIn("s96_taiga_originalBPE_offline_UD_targets",
                         p.read_text())


if __name__=="__main__":
    unittest.main()
