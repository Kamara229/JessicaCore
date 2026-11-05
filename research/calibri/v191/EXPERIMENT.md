# CALIBRI v191 — read-only format-error audit of frozen v190 replies

v190 actually completed under GitHub CI #42092017343. Its
conservative whole-answer exact-string grader recognized only
9/64 for original 135M and 18/64 for original 360M;
47/64 and 38/64, respectively, were **UNGRADED**.
These ungraded answers include explicit arithmetic equations
(e.g., `200 + 37 = 237`) and ownership sentences (e.g.,
`NeriDalo0's folder is purple.`). This is a genuine metric
problem: a valid equation or subject-qualified sentence can
be substantively correct without being exactly a bare number
or color. Conversely, a wrong calculation has a perfectly
well-formed equation, and simply detecting the gold word in
an arbitrary sentence is unsafe.

This is a **retrospective exploratory** and **read-only**
analysis on the 128 *existing* original source answers,
not a NEW unbiased test, an independently locked evaluator,
or a compressed model. The original v190 grader results
are preserved unchanged in the report as primary preregistered
strict scores; never relabel them as originally verified.

Implement narrow audited exact-full-response recognition of:
1) arithmetic `a + b = c`, `a - b = c`, `a minus b equals c`
   where the explicitly stated operands match the question;
   require the whole equation to be one claim;
2) answer about the **specific named original person**
   from the question via unambiguous possessive
   `Person's folder is [a] color` or
   `Person owns the [color] folder`; not mere gold words
   quoted in any unrelated context;
3) carefully recognized full factual assertions with a single
   unambiguous answer, no contradictions or negations;
4) all original strict Yes/No bare answers remain unchanged.

Retain independent `machine_wrong` where the asserted
answer is recognizably wrong, and `ungraded` for
unknown/contradictory/negated/unsupported texts.
Test with adversarial examples including an incidental
correct word followed by incorrect assertion; preserve
raw texts and both grades for audit. No source editing,
training, codebook optimization, new model inference,
PRIVATE TEST, LIVE Jessica, main, Render, Supabase, APK,
Green Tunnel, or Jessica-run test.

**Scientific next gate**: any revised evaluator must be
committed BEFORE a further entirely new frozen question
set to be called an independent confirmation. v191
numbers are post-hoc characterization of annotation error,
not valid independent generalization.
