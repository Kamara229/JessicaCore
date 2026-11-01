# Codex CALIBRI: long-session screen and idle gesture verification (2026-11-01)

## Owner's objective

Jessica Codex should take the owner's place in a deliberately selected ChatGPT thread for backend or isolated CALIBRI research, without stopping after each completed research stage. It must not claim unperformed experiments or change production resources or source code through the Android phone.

The owner also requested a small scroll up then down every three minutes of ChatGPT work to prevent the phone entering sleep.

## Two independent Android mechanisms

**1. Screen-on request**: the foreground Accessibility Overlay for Codex already supplies `WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON`. The general WindowManager API calls this a window flag; current Android's "Keep the screen on" guide recommends setting it *in an Activity, never directly in a Service*. The experimental overlay cannot be claimed to keep the owner's actual Android/Xiaomi screen awake until tested on that physical device. On a timeout shorter than three minutes, the swipe alone cannot prevent sleep. Codex refuses UI activity on a noninteractive or locked phone; the owner Stop button remains topmost.

**2. Optional 180-second micro-scroll**: a separate, explicitly owner-enabled session-local beta pulse is now implemented. It is **OFF by default**, resets on every restart, and runs only when all of the following are true:
- ChatGPT is the selected foreground app, the user's Codex session remains active and selected, no lock/sleep and no pending interpretation.
- The app is actually generating a reply; Codex is waiting for that same reply. Never in a composer preparation/send path.
- The current reply's exact unique `JESSICA_END_<nonce>` marker is present within a central, visible, enabled, scrollable transcript subtree. Its bounds must safely exclude toolbars, sidebar and composer; geometry alone is not sufficient. No verified accessibility node => no touch.
- At least 180,000 ms elapsed after both the outgoing send and the previous gesture attempt.
- The touch comprises two separated **16dp movements** in opposite directions inside the validated center of the transcript (32dp up plus 32dp down of finger travel, 550 ms total). No click, navigation or send is introduced. If Android cancels/rejects it, further pulses automatically disable; no immediate retries.
- After any attempt, outgoing sends require a fresh visible confirmation of the previously completed reply's exact marker. If this proof disappears, the relay stops sending rather than switching chats.

The existing `CodexGestureScroller`, `CodexScrollFollower` and their general automatic scrolling entry points remain hard-disabled. This beta cannot recover replies whose markers are outside the viewport.

## Before considering a production release

**MUST RUN ON THE OWNER'S ACTUAL DEVICE**; GitHub emulator/build tests cannot establish safety, display-timer behavior, or ChatGPT native UI compatibility.

1. Install the reviewed *debug* candidate with no production merge. In Android display settings choose a short timeout (e.g. 30 or 60 seconds); do not turn on battery saver.
2. Open an innocuous test chat in the official ChatGPT app, start Codex, select the chat and check Stop remains usable. With beta **OFF**, observe >5 minutes without touching. Record if the display remains interactive; if it sleeps, overlay `KEEP_SCREEN_ON` is insufficient on this device.
3. Turn the experimental beta **ON** while deliberately requesting a sufficiently long ChatGPT answer. Watch it reach 180 seconds and check that the 32dp up/down stroke stays within the same transcript, doesn't enter sidebar, doesn't change the draft, and doesn't break the Stop button. If the reply ends early, the gesture correctly does not run.
4. Check Stop immediately interrupts continued work. Check lock, screen-off, incoming notification/shade, keyboard, unrelated activity, missing editor, and switching to another chat: there must be zero gestures and zero sends. Check that cancellation disables the pulse instead of repeating it.
5. Validate a 30/60 second screen timeout independently from the 180-second gesture. If the display sleeps first, no claim of awake preservation is allowed. Use manual longer timeout/charging while designing a supported screen-on route.

**Status:** isolated draft PR only. Opt-in gesture implementation and pure policy tests are committed; actual real-device safety and sleep prevention are UNVERIFIED. Do not enable by default, merge to main, deploy Render/Supabase, or install on the owner's phone without successful device testing and review.

Sources:
- https://developer.android.com/develop/background-work/background-tasks/awake/screen-on
- https://developer.android.com/reference/android/view/WindowManager.LayoutParams
- https://developer.android.com/guide/topics/ui/accessibility/service
