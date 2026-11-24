import unittest,copy
from research.calibri_nra3.s119q2i8z67_eval_guard import verify_pins,results,TRAIN_ARMS,CHILD_SHA,GT_SHA,ARMS,rows
class TestZ67(unittest.TestCase):
    def fixture(self):
        weights={a:{"joint":"a"*64,"GT12":"b"*64} for a in TRAIN_ARMS}
        report={"stage":"S119Q2I8Z65_GPT_JOINT98_PAIR_EOS_UTF8","Z55_joint_SHA256":CHILD_SHA,"Z55_GT12_SHA256":GT_SHA,
           "FP32_calls":0,"no_DEV_scoring":True,"total_actual_training_forwards":3236,
           "total_actual_replay_forwards":1684,"total_Adam_updates":196,"output_gate":0,"candidate_accepted":False,
           "saved_research_only":{a:{p:{"sha256":s} for p,s in w.items()} for a,w in weights.items()},
           "training":{a:{"Adam_updates":98,"native_training_forwards":1618,"GT12_learning_rate":.0015,
             "GT12_abs_max_history":[.4],"FP32_calls":0,"UTF8_loss_positions":1618 if a==TRAIN_ARMS[1] else 0} for a in TRAIN_ARMS}}
        return report,{"run_id":45435892186,"conclusion":"success","head_sha":"b80cb13b4d056a716be3fb5f55c34a2cd82836ff","weights":weights}
    def test_no_partial_or_unpinned(self):
        r,p=self.fixture();verify_pins(r,p)
        for bad in ("conclusion","weights"):
            q=copy.deepcopy(p);q[bad]="failure" if bad=="conclusion" else {}
            with self.assertRaises(ValueError):verify_pins(r,q)
        r["training"][TRAIN_ARMS[1]]["GT12_abs_max_history"]=[.5]
        with self.assertRaises(ValueError):verify_pins(r,p)
    def test_FP32_not_gate(self):
        def data(n):
            return [{"id":r["id"],"generated_text":"x","text_correct":False,"complete_correct":False,"EOS":False,"valid_UTF8":i<n} for i,r in enumerate(rows())]
        rec={a:data(0) for a in ARMS}
        rec[ARMS[2]]=data(4)
        a=results(rec);rec[ARMS[3]]=data(42)
        self.assertEqual(a["exploratory_gate"],results(rec)["exploratory_gate"])
        self.assertTrue(a["all_exploratory_gates_pass"])
        self.assertFalse(a["candidate_accepted"])
if __name__=="__main__":unittest.main()
