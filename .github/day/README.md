# Jessica Day Shift — without paid model APIs

This is an **API-free deterministic self-audit**, not the generative Codex model.

In the Android drawer open Codex → **☀ Дневной режим**. Choose Auto, then
enter the owner key and Start. The backend dispatches
`.github/workflows/jessica-day.yml`. You can close the phone: GitHub Actions
continues independently, and the latest session status is visible later.

**One-time setup:** configure `JESSICA_CODEX_OWNER_KEY` and
`JESSICA_GITHUB_AUTOMATION_TOKEN` in Render exactly as in the legacy Codex
README. Do **not** configure OPENAI_API_KEY; the day workflow does not
reference it. The GitHub token needs repository Issues and Actions scopes.
The owner key is not stored on the phone. Never put secrets in public issues.

The bounded workflow checks source syntax, runs pinned regressions, and if a
known regression recipe is still absent, creates **one** new meaningful
test and a reviewable PR. It never auto-merges or auto-deploys and does not
call Groq/OpenAI or modify Experience/Skills.

Goals: Auto (first unmet recipe), Language, Context, Foundations. After
all recipes are installed a run audits and reports **no change**. It does not
invent autonomous reasoning or claim unsupervised code learning. The next
stage would require an explicitly connected **free/on-device** model and
verified execution sandbox to safely propose open-ended improvements.

GitHub Actions uses standard hosted runners; their pricing/allowances depend
on whether a repository is public or private. See GitHub Actions billing.
A single run is bounded to 18 minutes. Runs must be owner-authenticated.
For private repos, check your Actions minutes quota before use.

The older API-billed Codex workflow is **unchanged**, kept separate and opt-in.
