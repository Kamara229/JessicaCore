# Jessica Codex: participant in the selected ChatGPT conversation

The owner prepares a backend improvement plan in ChatGPT, starts Codex in
Jessica, opens the desired conversation, and presses **Чат выбран** on the
overlay. Jessica first sends **Джессика готова к работе в чате.** and explains
her role. Both Chat and Work conversations may be selected. ChatGPT uses its
own available GitHub, Render and Supabase connections; selecting a chat does
not grant it connections that it does not already have.

Jessica itself does not create commits, pull requests, merges or deployments.
The Codex HTTP router exposes only authenticated `/api/codex/relay/interpret`.
Old phone/day/paid/candidate routes are no longer registered. Historical
candidate modules and manually dispatched validation workflows are not part
of the new runtime. No owner key, goal selector, personal chat URL or separate
paid OpenAI API is used. Understanding uses Jessica's existing Groq backend;
that provider's availability and usage limits still apply.

## Setup and operation

Enable **Jessica — Кодекс** in Android accessibility settings and accept the
Codex disclosure. The existing service component name is preserved on upgrade.
Leave the phone unlocked, powered and displaying the selected ChatGPT chat.
Do not change conversations during a run. Leaving ChatGPT clears confirmation;
returning requires **Чат выбран** again. The app cannot reliably obtain a
ChatGPT internal conversation ID. Follow-up messages require the preceding
session marker in the selected conversation before composing.

The initial visible plan, recent session messages and real prior test results
are sent to Jessica's authenticated interpreter. It chooses a contextual reply,
one exact test prompt, completion, or a pause for the owner. It cannot execute
tools, edit infrastructure, invent test results, or authorize owner-only actions.
Tests use the same `JessicaAIEngine.solve` path as Jessica's normal text chat.
Each new test gets its own stored chat; explicitly requested follow-ups reuse
the preceding test's history. Requests and verbatim results are visible in the
normal chat list alongside a Codex conversation journal. HTTP request success
is reported separately and is never called a passing test.

Jessica remains in ChatGPT while its ordinary chat engine executes the test;
opening Jessica's chat screen is unnecessary. The entire actual result is
returned to ChatGPT for evaluation. Oversized results are kept locally and
pause the session rather than being silently truncated. When no test is needed,
the interpreter replies using the plan and conversation context. It stops at
completion or when an action outside its role needs the owner.

## Reliable reading and stopping

Each outgoing message asks ChatGPT to put its natural-language answer between
two unique, standalone session markers. No JSON is required in ChatGPT.
Only a complete, stable response for the current turn is accepted. Incomplete
or missing markers cause waiting and ultimately a timeout, not a guessed answer.
The accessibility adapter excludes the editor and its descendants. It accepts
only the official ChatGPT package, a unique editable field and exact send
labels. It does not tap arbitrary coordinates or inspect other apps.

Existing drafts are never overwritten. The send is recorded before clicking;
an ambiguous click or unexplained editor change is never automatically retried.
A visible already-posted request is not inserted again. Stop, screen lock,
service disconnection, timeout or uncertain interpretation ends the session.
Stop cannot undo a test already received by Jessica's backend or work already
performed by ChatGPT. No session automatically resumes after process restart.
One session is limited to eight hours, sixty sends and twenty minutes per
ChatGPT answer. Session history is stored in Jessica's private chat storage and
can be managed through the ordinary chat list.

## Validation

- `node --test backend/scripts/regressionCodexRelay.test.js` checks contextual
  interpretation inputs, exact test extraction, malformed decisions and the
  absence of infrastructure-writing routes.
- `gradle testDebugUnitTest assembleDebug` checks reply framing, send limits,
  draft protection and the existing Android regression suite.
- `gradle connectedDebugAndroidTest` verifies framework service binding and
  actual local test-chat persistence, verbatim errors, contextual continuation,
  follow-up history and stopping without test execution, using injected model
  responses. It does not call a paid model or assume tests passed.

A successful build/emulator test does not prove compatibility with the owner's
current native ChatGPT UI. On the phone, verify explicit selection, one readiness
send, full response capture, one requested test with its real report, contextual
continuation, and Stop. Long or inaccessible UI responses can still cause a
reported pause and must be validated on that device.
