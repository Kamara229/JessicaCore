# Jessica ↔ Codex autonomous engineering mode

This is **two independent software agents**, not a direct connection to a specific ongoing ChatGPT conversation.
Jessica uses its server-side AI provider for peer reviews, while Codex executes in GitHub Actions under the OpenAI Codex GitHub Action.

## One-time owner-only setup

1. In **Render** service `JessicaCore`, configure **secret environment variables**:
   - `JESSICA_CODEX_OWNER_KEY`: unique random passphrase, at least 24 characters; do not commit it or embed in Android.
   - `JESSICA_GITHUB_AUTOMATION_TOKEN`: fine-grained GitHub token issued by repo owner `Kamara229`, restricted to **JessicaCore** with **Issues (read/write)** and **Actions (read/write)** permissions; repository metadata read is implicit.
2. In GitHub **JessicaCore → Settings → Secrets and variables → Actions**, add `OPENAI_API_KEY` to the repository secrets. This is a separately billable **OpenAI API key**, not a ChatGPT subscription login.
3. Verify the already configured `JESSICA_APP_TOKEN` GitHub Actions secret still exists.
4. In GitHub **Settings → Actions → General**, permit workflows to create pull requests if the repository requires it. Do not grant the agent permission to auto-merge.
5. Install the new APK **over** the existing Android installation. In drawer open **Codex ↔ Jessica**, enter your private **owner key** (kept in Android RAM only), pick **language**, **context** or **foundations**, and tap **Start**.

You can observe the role-labelled agents' conversation in the Android screen, which polls GitHub issue comments via the backend. An open session issues a single plan + Jessica review + implementation + Jessica review. No personal message history or private documents are sent to these issues; this is a **public repository**, and issue conversations are public. The owner can stop; the server also requests cancellation of active GitHub Actions runs. Cancellation can take effect after the current API call or step.

## Security invariants

- The standard `JESSICA_APP_TOKEN` embedded in Android cannot authorize privileged start, status or stop operations alone.
- The owner key never enters the GitHub workflow, repository, build config or logs.
- No arbitrary natural-language prompt or repository URL is accepted from Android: the objective is one of three fixed allowlisted goals.
- Agent modifications are restricted to `backend/core/{language,chat,foundations}/*.js` and `backend/scripts/regression*.test.js`. No auth, production DB, CI, keys, billing or deployment writes are allowed.
- Each work session is bounded to two Codex turns, two Jessica reviews, a strict changed-file guard and fixed regression tests.
- Generated code cannot be merged or deployed automatically. A **pull request** is opened on a new branch for inspection; the release pipeline remains separate.
- GitHub Actions jobs use disposable runners, checkout without persisted credentials and the Codex Action with strict permission profiles.
- An OpenAI API key is required for Codex. Running it incurs API usage charges. No Codex runs are started without the owner's explicit **Start**.

## Troubleshooting

- `codex-not-configured`: configure Render environment variables.
- `owner-key-required`: enter the same owner key as in Render.
- `codex-dispatch-failed`: verify fine-grained token scopes and workflow availability on `main`.
- GitHub Action fails at Codex: confirm `OPENAI_API_KEY` exists and has usable API billing.
- PR creation failed: allow GitHub Actions to open pull requests in repository settings.
- If a workflow fails, view the issue and its linked Actions run; the issue is closed after cleanup to allow a new session.
