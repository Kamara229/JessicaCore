import unittest
from types import SimpleNamespace
from v132.protocol import BANDS, band_of, conditions, all_examples, verify_roles
from v132.analysis import summarize


class Tests(unittest.TestCase):
    def test_fixed_non_overlapping_bands(self):
        self.assertEqual(len(BANDS), 6)
        self.assertEqual(len(list(all_examples())), 16)
        for depth in range(30):
            self.assertEqual(band_of(f"model.layers.{depth}.mlp.down_proj"), depth // 5)
        choices = dict(conditions())
        self.assertEqual(len(choices), 14)
        for n in range(6):
            self.assertTrue(choices[f"only_band_{n}"](n))
            self.assertFalse(choices[f"drop_band_{n}"](n))
            for other in range(6):
                if other != n:
                    self.assertFalse(choices[f"only_band_{n}"](other))
                    self.assertTrue(choices[f"drop_band_{n}"](other))

    def test_expected_layer_roles(self):
        roles = {f"model.layers.{i}.{kind}":object() for i in range(30)
                 for kind in ("self_attn.o_proj","mlp.down_proj")}
        verify_roles(SimpleNamespace(blocks=roles))
        roles.pop(next(iter(roles)))
        with self.assertRaises(ValueError):
            verify_roles(SimpleNamespace(blocks=roles))

    def test_summary_matches_anchors(self):
        ids = [(f"p{i//2}", ("left" if i%2==0 else "right")) for i in range(16)]
        def row(i,eos,margin):
            return {"pair":ids[i][0],"side":ids[i][1],
                    "eos_first":eos,"eos_margin":margin,
                    "eos_probability":float(eos),"best_expected_rank":5}
        arms = {"head_only":[row(i,False,-1.0) for i in range(16)],
                "complete":[row(i,True,1.0) for i in range(16)]}
        for band in range(6):
            arms[f"drop_band_{band}"] = [row(i,False,-1.0) for i in range(16)]
            arms[f"only_band_{band}"] = [row(i,False,-1.0) for i in range(16)]
        z = summarize(arms)
        self.assertEqual(z["by_arm"]["complete"]["first_eos_argmax"], 16)
        self.assertEqual(z["by_band"]["0"]["drop_band_reverses_full_EOS"], 16)
        arms["complete"][0]["side"] = "right"
        with self.assertRaises(ValueError):
            summarize(arms)


if __name__ == "__main__":
    unittest.main()
