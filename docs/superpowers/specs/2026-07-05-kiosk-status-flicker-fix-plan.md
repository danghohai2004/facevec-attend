# Fix plan: kiosk status bar flickers on/off

**For:** implementing agent (GPT 5.5)
**Scope:** frontend only. No backend, no new deps, no refactor.

## Problem

The `/kiosk` status bar (`frontend/src/components/kiosk/kiosk-screen.tsx` → `StatusBar`)
blinks its color/text rapidly while a face is present. One bar is driven by three
signals that each update on a different clock:

- `proximity` — recomputed **every animation frame (~60fps)** in `use-face-tracker.ts`
- `hint` — from backend WS messages, ~1/sec
- `shiftWindow` — ~1/min

Whichever changes, the bar changes. The fast blink comes from `proximity`.

## Root cause (primary — fixes the fast blink)

`frontend/src/components/kiosk/use-face-tracker.ts`, the no-detection branch (currently ~lines 82-88):

```ts
if (!dets || dets.length === 0) {
  hideBox();
  if (canCaptureRef) canCaptureRef.current = false;
  wasClose = false;
  setProximity("none");
  return;
}
```

MediaPipe drops a detection for a frame or two even while a face sits still.
Each dropped frame flips `proximity` → `"none"` (amber "Đưa khuôn mặt lại gần"
disappears, falls back to emerald base) then the next frame flips it back.
Many times per second = flicker.

The existing hysteresis (commit `149f21e`, `MIN/KEEP_FACE_AREA_RATIO` in
`lib/kiosk.ts`) only smooths **ok↔far** by area. It does **not** cover
**present↔none** (detection dropouts). That gap is the bug.

## Fix (primary)

Add a time-based grace window: don't clear to `"none"` until the face has
genuinely been absent for a short interval. Same anti-flicker intent as the
existing area hysteresis.

In `use-face-tracker.ts`:

1. Add module const near the type exports:
   ```ts
   const LOST_GRACE_MS = 500; // hold last state through brief detection gaps
   ```
2. Inside the effect, next to `let wasClose = false;`:
   ```ts
   let lastSeen = 0; // performance.now() of the last frame with a detection
   ```
3. Replace the no-detection branch so it returns early during the grace window:
   ```ts
   if (!dets || dets.length === 0) {
     if (performance.now() - lastSeen < LOST_GRACE_MS) return;
     hideBox();
     if (canCaptureRef) canCaptureRef.current = false;
     wasClose = false;
     setProximity("none");
     return;
   }
   ```
4. When a detection IS found, record the time. Add right after the no-detection
   branch (before picking the largest detection):
   ```ts
   lastSeen = performance.now();
   ```

Note: `raf = requestAnimationFrame(loop)` is already called (~line 80) before
this branch, so the early `return` does not stop the loop — do not re-schedule.

Effect: during the 500ms grace the box, `canCaptureRef`, and proximity keep
their last value, so a 1-2 frame dropout no longer blinks the bar. Harmless
side effect: capture may fire for up to ~500ms after a face truly leaves.

## Secondary (optional — slower 1Hz blink)

Only do this if the bar still blinks ~1/sec after the primary fix, with a real
face present but unrecognized.

`frontend/src/lib/kiosk.ts`, `reduceKiosk`: `unknown` sets the red hint,
`no_face`/`spoof` clear it (`hint: null`). If the backend alternates
`unknown`/`no_face` at the recognition margin, the red bar blinks ~1/sec.

Cheapest option: on `no_face`/`spoof`, keep the existing `hint` instead of
nulling it (only `recognized`, `greeting_done`, or a fresh `unknown`/`error`
change it). Confirm this doesn't leave a stale red bar after the person walks
away — `greeting_done` and the proximity→`none` path must still clear the view.
If that tradeoff is unclear, leave it and report back rather than guessing.

## Verification

- `cd frontend && npm test` — existing `kiosk.test.ts` (reducer + `isFaceCloseEnough`)
  must still pass. If you touch the reducer for the secondary fix, add a case
  proving `no_face` after `unknown` keeps the hint.
- `npm run build` / typecheck clean.
- Manual: run the kiosk, stand at mount distance, hold still — the bar must hold
  steady, not strobe. Walk away — bar returns to base within ~0.5s.

## Do NOT

- Add a debounce wrapper around the whole `StatusBar` (would delay the green
  "recognized" greeting — that one must appear instantly).
- Touch backend, capture interval, or greeting timing.
- Add libraries.
