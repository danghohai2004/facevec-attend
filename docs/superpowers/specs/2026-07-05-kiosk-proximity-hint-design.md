# Kiosk: drop spoof warning, add "move closer" proximity hint

**Date:** 2026-07-05
**Status:** Approved (design)
**Scope:** `frontend/src/components/kiosk/` + `frontend/src/lib/kiosk.ts`

## Problem

Two changes to the kiosk attendance terminal:

1. The `spoof` anti-spoofing status shows **"Vui lòng nhìn thẳng vào camera"**.
   This is confusing to genuine users who get momentarily flagged. Remove it.
2. Today, when a real face is detected but is **too far** to capture, the screen
   shows the same neutral "đưa khuôn mặt vào khung" guidance as when there is no
   face at all — no actionable feedback. We want a distinct prompt telling the
   user to **move closer**.

## Current behavior (as-is)

- `use-face-tracker.ts` runs MediaPipe every animation frame. Per frame it writes
  `canCaptureRef.current` (a plain **ref**, not React state, so no re-render per
  frame) to one of:
  - no detections → `false`, box hidden
  - face detected, `isFaceCloseEnough(bb, fw, fh)` → `true`/`false`, box shown
- `use-recognition.ts` reads `canCaptureRef` and only sends a capture when `true`.
  So while a face is **too far**, no capture is sent → no backend hint arrives.
- `kiosk.ts` reducer maps backend statuses to `hint`. `spoof` → `"Vui lòng nhìn
  thẳng vào camera"`.
- `StatusBar` (`kiosk-screen.tsx`) renders one bottom bar with priority:
  `recognized` → backend `hint` → shift-window guidance → out-of-hours.

The proximity signal already exists; it is simply not surfaced to the UI because
it lives in a ref.

## Design

### 1. Remove the spoof warning (`kiosk.ts`)

In the reducer's `message` handler, change the `spoof` case to set `hint: null`
(mirror the existing `no_face` case), keeping `faceBox: null`. Result: a spoof
result renders no warning and falls through to normal shift-window guidance.

```ts
case "spoof":
  return { ...state, hint: null, faceBox: null };
```

### 2. Surface proximity as React state (`use-face-tracker.ts`)

Add a tri-state React state alongside the existing `canCaptureRef` writes:

```ts
type Proximity = "none" | "far" | "ok";
const [proximity, setProximity] = React.useState<Proximity>("none");
```

Set it at each point the loop already decides proximity, so it stays in lockstep
with `canCaptureRef`:

- no detections → `setProximity("none")` (alongside `canCaptureRef.current = false`)
- face detected → compute `close = isFaceCloseEnough(bb, fw, fh)` once, then
  `canCaptureRef.current = close; setProximity(close ? "ok" : "far");`

**Perf:** `setProximity` is called every frame, but `useState` bails out of
re-rendering when the next value is `Object.is`-equal to the current one, so a
steady state (held "far", held "ok") causes zero extra renders — only genuine
transitions re-render. This preserves the ref's original purpose (no per-frame
renders) while making transitions visible to the UI.

**Failed / loading tracker:** on init failure, inference failure, or before the
detector is ready, do **not** force proximity to any capture-blocking value.
Leave `proximity` at its last value (starts `"none"`) and keep the existing
`canCaptureRef.current = true` fallback. The UI only reacts to the explicit
`"far"` state (see §3), so a machine with no WebGL behaves exactly as today.

Return `proximity` from the hook. Since it currently returns a bare
`TrackerStatus`, return an object to avoid a lossy signature:

```ts
return { status, proximity };
```

Update the one caller in `kiosk-screen.tsx` accordingly.

### 3. Render the "move closer" bar (`kiosk-screen.tsx`)

Destructure `{ status: trackerStatus, proximity }` from `useFaceTracker`. Pass
`proximity` into `StatusBar`. New priority order in `StatusBar`:

1. `recognized` + greeting → green (unchanged)
2. `scanning` + backend `hint` → red (unchanged)
3. **`scanning` + `proximity === "far"` → amber, "Đưa khuôn mặt lại gần hơn"** (new)
4. `scanning` + shift-window guidance (check_in / check_out / undefined) → unchanged
5. `scanning` + out-of-hours → unchanged

Amber styling to match the terminal's solid-bar convention: `bg-amber-600
text-white`, reuse the existing `AlertTriangle` icon block. `"far"` only ever
coexists with an absent backend hint (captures are suppressed while far), so
ranking it below `hint` is safe and never hides a real result.

Wording: **"Đưa khuôn mặt lại gần hơn"** (terse, uppercase, matches the bar).

## Out of scope / YAGNI

- No change to `isFaceCloseEnough` thresholds or the capture gate itself.
- No "too close" state — only "too far" was requested.
- No change to the server-box fallback path.

## Verification

- Manual: run the kiosk, stand far from the camera → amber "Đưa khuôn mặt lại
  gần hơn"; step closer → bar returns to shift-window guidance / recognition.
- A spoof result no longer shows "Vui lòng nhìn thẳng vào camera".
- Sanity check that holding a steady distance does not spam re-renders (React
  DevTools profiler or a temporary render counter) — confirming the `useState`
  bail-out holds.
