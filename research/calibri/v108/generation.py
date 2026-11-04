"""Real model generated 32-token text; no teacher activation injection."""
from __future__ import annotations
from hashlib import sha256
from v94.generation import generate_one
from v96.grade import grade_answer, string_breakdown
from v96.answer_review import first_response,leading_reference_signal
from v106.signals import generation_signals

def build_chat(tok,question):
    p=tok.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=False,add_generation_prompt=True)
    if not isinstance(p,str) or not p:
        raise AssertionError("original pretrained chat-template prompt missing")
    if question not in p:
        raise AssertionError("actual user question lost in chat formatting")
    return p

def assistant_turn(tok, generated):
    """Separate the real first turn from the tail of a fixed-work benchmark."""
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError("missing tokenizer end-of-turn ID")
    ended = eos in generated
    stop = generated.index(eos) if ended else len(generated)
    return {"text": tok.decode(generated[:stop], skip_special_tokens=True),
            "content_tokens": stop, "ended_by_eos": ended,
            "tokens_after_eos": len(generated) - stop - int(ended)}

def observed_answer(model,tok,category,question,gold):
    chat=build_chat(tok,question)
    data=generate_one(model,tok,chat)
    generated=data["ids"]
    if len(generated)!=32:
        raise AssertionError("not exactly 32 genuine generated output tokens")
    text=data["text"]
    turn=assistant_turn(tok,generated)
    grade=grade_answer(text,gold) if gold else None
    signal=leading_reference_signal(text,gold) if gold else None
    return {
        "domain":category,
        "original_question_SHA256":sha256(question.encode("utf8")).hexdigest(),
        "actual_chat_prompt_SHA256":data["prompt_SHA"],
        "original_pretrained_chat_template_rendered":True,
        "ground_truth_answers":list(gold),
        "actual_decoded_model_output":text,
        "first_assistant_turn":turn,
        "role_line_within_first_assistant_turn":generation_signals(turn["text"])["simulated_role_line_present"],
        "first_visible_response_before_fake_role_lines":first_response(text)[:300],
        "actual_32_generated_token_ids":generated,
        "full_model_real_CPU_KV_cache_elapsed_seconds":data["elapsed_seconds"],
        "v106_frozen_unchanged_morphology":generation_signals(text,generated),
        "v96_first_response_prefix_gold_signal":grade,
        "v96_first_sentence_gold_presence_not_semantic_proof":signal,
        "token_lexical_observations":string_breakdown(text,generated),
        "requires_independent_human_grammar_and_factual_review":True,
        "no_automatic_semantic_correctness_claim":True,
    }
