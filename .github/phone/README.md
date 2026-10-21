# Ordinary ChatGPT on an unlocked Android phone

This is an experimental, explicitly started UI assistant for the official
`com.openai.chatgpt` app. It does not use paid model APIs, browser cookies,
ChatGPT private HTTP endpoints, credentials from ChatGPT, or Work mode.
ChatGPT subscription limits still apply. Live UI compatibility must be
tested on the owner's device; a successful Android build does not establish it.

## One-time setup

Use the existing Render settings `JESSICA_CODEX_OWNER_KEY` and
`JESSICA_GITHUB_AUTOMATION_TOKEN`. The GitHub token must be able to read the
repository and Actions results, create Git data/branches and pull requests,
and merge PRs if the owner elects automatic application. GitHub Actions must
be enabled. No OpenAI/Groq API key is needed by this path. The shared app
token embedded in an APK is never accepted as owner authorization.

After installing the updated signed APK, open Codex → **Обычный чат**.
Enable **Jessica — обычный ChatGPT** in Android accessibility settings.
Read the data-access disclosure, enter the independent owner key, choose
a goal, and start. An optional private `/c/<uuid>` URL opens the selected
conversation; otherwise choose a dedicated ordinary chat after ChatGPT opens.
The owner key is held only in process memory and is not included in prompts.

Leave the phone unlocked, powered and displaying ChatGPT. Select **Чат**,
then press the overlay **Чат выбран — начать**. This is an explicit user
attestation of the visible mode when the app version does not expose a
selected-mode label to accessibility. An observed Work/Codex mode blocks
the sender. Leaving ChatGPT resets the attestation. Arbitrary coordinates
and fuzzy button matching are not used. If the editor or Send control is
ambiguous, the assistant waits and reports the problem; it does not guess.

## A bounded run

1. Load two public source files at an exact main commit.
2. Send one engineering request in the selected ordinary chat. At most two
   format-repair follow-ups are allowed. Existing drafts are never overwritten.
3. Read only the nonce-delimited answer for this launch after it is stable.
   The complete UI transcript is not saved or sent to the server.
4. Validate exact file paths, original blob SHAs, size, and a new regression.
5. Open one source-only PR. No chat text is posted to GitHub.
6. CI from the trusted base executes the regression in network-disabled,
   read-only, unprivileged containers with no secrets. The new test must
   demonstrate an assertion failure on the original source, and then pass
   with the correction alongside the fixed base regression suite.
7. If the owner checked automatic application before launch, an exact-head
   success may merge the PR, only while main still equals the reviewed base.
   Existing Render/APK deployment workflows then run normally. Deployment
   success is not claimed from a merge result. Otherwise the verified PR waits.

The overlay Stop button cancels UI activity and requests server cancellation.
Already submitted GitHub merges cannot be undone by Stop. An interrupted
network write is reported as uncertain, never retried as a new operation.
Server restart, screen lock, response timeout, changed base, failed tests or
unrecognized UI stop or pause the run. The run is limited to one correction,
three messages and one hour; it is not an indefinite background loop.

Only the last ten nonce-delimited answers and the latest status/PR link are
kept in private application storage. Owner keys are not persisted. No positive
Experience or Skill is fabricated from model text, PR creation, or build success.
This change improves source through verified commits; it does not train model weights.

## Validation

`node --test backend/scripts/regressionPhoneChat.test.js`

`gradle testDebugUnitTest assembleDebug`

Before relying on unattended operation, the device test must verify ordinary
mode selection, one send (no duplication), full answer capture, Stop, screen
lock, and a failed-test candidate remaining unmerged. Unknown UI layouts are
unsupported until an adapter is validated on the actual screen.
