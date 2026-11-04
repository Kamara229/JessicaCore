"""Original-model chat prompt and label-prefix parsing contract."""
from __future__ import annotations
import unittest
from unittest.mock import patch
from v103.spec import BOOLQ_RANGE,ASSERTIONS
from v103.qa import chat
from v103.text import matched_generation

class Tok:
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        if not add_generation_prompt or tokenize:
            raise AssertionError("chat generation prompt was not requested")
        return "<user>"+messages[0]["content"]+"<assistant>"

class Tests(unittest.TestCase):
    def test_user_role_generation_prompt_is_preserved(self):
        self.assertEqual(chat(Tok(),"Count"),"<user>Count<assistant>")
    def test_questions_are_new_original_train_slice(self):
        self.assertEqual(BOOLQ_RANGE,(8780,8804))
        self.assertEqual(len(ASSERTIONS),4)
        self.assertEqual(len(set(x[1] for x in ASSERTIONS)),4)
    def test_strict_first_value_is_not_arbitrary_substring(self):
        saved={"ids":[1,2],"text":"13 is the result.",
               "full_model_token_steps":2,"elapsed_seconds":.01,
               "prompt_SHA":"abc"}
        with patch("v103.text.generate_one",return_value=saved):
            ok=matched_generation(None,Tok(),"Count","13")
            self.assertTrue(ok["exact_label_prefix_matches_truth"])
        saved["text"]="The answer may be 13, perhaps."
        with patch("v103.text.generate_one",return_value=saved):
            bad=matched_generation(None,Tok(),"Count","13")
            self.assertFalse(bad["exact_label_prefix_matches_truth"])

if __name__=="__main__":unittest.main()
