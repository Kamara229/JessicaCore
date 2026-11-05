# CALIBRI v192 — freeze a safer QA grader before NEW answers (2026-11-05)

**Scope**: deterministic, source-independent, standard-library answer
recognizer + three-way grading, not a model experiment. Previous
v190 and v191 responses and prior scores must remain unchanged.

## Evidence motivating freeze

- v190 original uncompressed FP32 135M/360M 64 novel public DEV
  questions: original frozen grader confirmed **9/64 vs 18/64**,
  **47/64 vs 38/64 ungraded**. Those are the original correct
  preregistered metrics; a large fraction is ambiguous due solely
  to the overly strict requirement for a single bare answer.
- v191 successful [run #42093390298](https://github.com/Kamara229/JessicaCore/actions/runs/42093390298)
  audited the exact old 128 answers using a *RETROSPECTIVE*
  narrow syntax recognizer: **34/64 vs 43/64** classified
  correct post-hoc. This was after seeing the raw answers,
  so it is **NOT** independent QA confirmation.
- Both FP32 original models produced **0/8** fully correct
  reversed Yes/No pairs. 360M cannot be assumed universally
  better than 135M on fictional binding. No compressed model
  evaluated in v190/v191/v192.

## Frozen scoring contract (no ML grader)

Inputs: frozen public QA case with immutable case_id, domain,
question, GOLD answer and optional pair ID; exact original
verbatim generated text from the checkpoint's own tokenizer.

Outputs: exact raw answer, machine_confirmed,
machine_recognized_wrong, or **ungraded_not_known_wrong**,
recognized single answer only when the complete utterance
matches a narrow predeclared language form. A gold word
mentioned inside an unrelated explanation, a negation, a
multivalued answer, a wrong person's color or an equation
with different operands is **never promoted** as correct.
Only one single complete claim can be machine-correct.

- Arithmetic: one integer or one fully bound original
  plus/minus/times expression and computed integer.
- Contradictory Yes/No: exactly one response Yes or No
  (possibly with a neutral lead-in), and **both halves**
  of opposing pairs must be correct to score a pair.
- Fictional binding: single bare color or a statement
  regarding exactly the named *queried* owner in the
  original question.
- Factual short: one valid single-term answer or a
  narrowly predeclared full subject-bound fact claim;
  matching the *same country and same chemical symbol*
  is required for assertion recognition.
- Long, multi-clause, multiline, contradictory, negated,
  or unknown answers are ungraded when they cannot be
  safely recognized. No semantic correctness by a
  second LLM (which would be biased/uncontrolled).

Separate source-of-recognition labels allow auditing
mistakes without promoting candidate gold mentions.

## Acceptance gate for future v193 (not launched here)

Run standard-library adversarial unit tests and produce a
SHA256 manifest for this version's source files in GitHub
Actions. Those hashes and all scoring decisions MUST be
locked **before** even constructing new v193 model outputs.
A future v193 experiment must make independent new
questions and precommit golds and hashes before loading
the models. Preserve all raw outputs from both originals,
including incorrect and ungraded responses, actual paired
opposite behavior. v192 scores are *parser metrics*,
not human-reviewed semantic accuracy.

There is no new model test, training, compressed source,
native CPU performance, physically measured BPP, original
production deployment or Jessica text test in v192.
The sealed private v89 TEST stays unread.
Research-only unmerged PR, main/LIVE Jessica, Render,
Supabase, APK, Green Tunnel unchanged.
