# Codex stand-in for CALIBRI and 3-minute idle protection — 2026-11-01

## User requirement

- Owner chooses an existing ChatGPT thread and starts Jessica Codex. Jessica continues the *same* thread instead of the owner, including the CALIBRI weight-creation/compression research.
- Research must use isolated research branches and honest real-world size/quality/speed reports. Jessica does not impersonate GitHub/Render/Supabase access, execute repo writes through mobile, or falsely claim tests were run.
- During sustained ChatGPT generation, prevent the Android display from timing out. Requested fallback: very short scroll up then down every 180 seconds.

## Shipped as a draft code change

- Existing ChatGPT accessibility overlay already sets `WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON`; its Stop button remains accessible. This is a **request**, not proof that Xiaomi/Android will honor it.
- The first conversation handoff now detects CALIBRI in the owner-selected chat and uses a short CALIBRI-specific message. The normal backend handoff is unchanged.
- Backend model instructions now allow CALIBRI work under the same no-direct-tools/no-secrets boundary.
- The relay refuses UI mutations when Android reports that the display is not interactive; the existing stop/lock/thread-verification controls stay in place.

## Safety-critical gap: gesture fallback remains disabled

The repository currently has `CodexScrollSafetyPolicy.mayScrollAutomatically() == false`, `CodexGestureScroller.scroll() == false`, and a disabled `CodexScrollFollower`. These locks were deliberately installed after an actual sidebar/new-conversation incident. Reenabling swipe dispatch without verifying that a specific node belongs to the selected transcript could tap another chat, hide the composer, or send into a wrong thread. Merely choosing a centrally located scrollable node is not proof.

Additionally, a 180-second gesture does **not** prevent a screen with a 30/60-second inactivity timeout from switching off before its first gesture. Android docs recommend `FLAG_KEEP_SCREEN_ON` on a foreground Activity, and the existing accessibility overlay flag requires **physical device** validation. Do not claim that `dispatchGesture` guarantees device wakefulness.

## Gate before enabling 3-minute up/down pulse

1. On an owner's physical phone, measure the screen state for >5 minutes with ChatGPT in foreground, Codex overlay on and a shorter device timeout; check whether the existing overlay flag alone holds the screen awake.
2. Establish a *content-based* transcript node identity, including the session's exact nonce, with no sidebar, sheet, keyboard, system shade or dialog. Reject ambiguous roots, wrong chat, partial identity, screen off, keyboard editing and user touch.
3. Implement a short, strictly bounded up+down gesture **only** inside that identified transcript, only when the current selected ChatGPT reply is actually generating, once every 180 seconds, not after completion or in a hidden window. Never retry a cancelled gesture.
4. Test the 3-minute pulse with actual ChatGPT versions: real-device Stop must work immediately, the same chat remains selected, no sends are duplicated, the field remains intact, no sidebar/new chat opens, and the display remains awake even with a shorter inactivity timeout. Abort if any gate fails.
5. User-visible kill switch and clear reporting when the screen-hold mechanism is not verified.

Until these gates pass, **no automatic scroll has been enabled**. This is an intentional fail-closed partial implementation: CALIBRI dialogue scope is implemented; physical sleep prevention and 3-minute gesture require device validation. Do not merge this draft to main or build/install an APK without a review.
