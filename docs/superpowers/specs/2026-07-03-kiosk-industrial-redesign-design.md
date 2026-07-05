# Kiosk Industrial Redesign

Date: 2026-07-03
Status: approved
Executor: external agent (GPT-5.5)

## Goal

Restyle the kiosk attendance screen from its current "cinematic/sci-fi" look
(floating rounded pills, backdrop-blur, gradients, amber HUD brackets) to an
industrial time-clock terminal look (ZKTeco/Hikvision style): solid opaque
bars, flat surfaces, square corners, high-contrast color-coded states.

**Render-only change.** All behavior — WebSocket recognition flow, MediaPipe
face tracking, TTS announcements, shift-settings polling, capture gating —
stays exactly as is.

## Scope

- Modify: `frontend/src/components/kiosk/kiosk-screen.tsx` (render/JSX/classes only)
- Do NOT modify: `use-recognition.ts`, `use-face-tracker.ts`, `frontend/src/lib/kiosk.ts`,
  anything in `src/` (backend)

## Layout

Fullscreen mirrored camera stays as the background layer (unchanged `<video>`
element). Two solid opaque bars are layered over it:

### Header bar (top)

- Solid `zinc-950` (fully opaque, no blur, no gradient), clear bottom border
  (e.g. `border-b-2 border-zinc-800`).
- Left: title `CHẤM CÔNG` — bold uppercase, plus a smaller subtitle
  `Hệ thống điểm danh khuôn mặt`.
- Right: clock — monospace tabular time (HH:MM, large) and weekday + date
  line below (`vi-VN` locale, same formats as current `Clock` component).
- Replaces the current top gradient and absolutely-positioned clock.

### Status bar (bottom)

One full-width solid bar (~80px tall, large uppercase bold text, centered)
replaces ALL current floating UI: the shift-window pill, the hint pill, the
guidance pill, and the greeting checkmark circle. Exactly one state is shown
at a time, priority order (highest first):

1. **Recognized** (phase `recognized`, ~5s): bright green (`bg-green-600`),
   white text: `✓ XIN CHÀO {NAME} — {MESSAGE}` where message comes from the
   existing `greeting` object (e.g. "Bạn đã chấm công vào ca hôm nay").
   Keep the existing `aria-live` and sr-only equivalence; TTS still speaks.
2. **Hint** (scanning with `hint` set — spoof/unknown/busy): solid red
   (`bg-red-700`), white text, warning icon, hint text uppercase.
3. **Scanning, inside check-in window**: solid dark emerald
   (`bg-emerald-700`): `→ GIỜ VÀO CA — ĐƯA KHUÔN MẶT VÀO KHUNG`
4. **Scanning, inside check-out window**: solid dark sky blue
   (`bg-sky-700`): `← GIỜ TAN CA — ĐƯA KHUÔN MẶT VÀO KHUNG`
5. **Scanning, outside any window**: neutral (`bg-zinc-900`), muted text:
   `NGOÀI GIỜ CHẤM CÔNG`

Shift window detection reuses the existing `currentShiftWindow(now, settings)`
helper and the existing 60s `useQuery` polling — no logic changes.

### Face brackets

Keep the 4-corner bracket overlay (both the imperative tracker box and the
`ServerFaceBox` fallback) but restyle: square corners (no `rounded-*`),
white (`border-white`) instead of amber.

### Full-screen overlays (camera error / initializing / disconnected)

Keep structure and copy. Restyle flat: fully opaque `bg-zinc-950` (drop the
`/90` transparency), square corners, no `animate-in` slide/zoom (a simple
fade or none is fine). Icons and text unchanged.

## Remove entirely

- Top and bottom gradient divs
- All `backdrop-blur-*` classes
- All `rounded-full` / `rounded-2xl` pill styling in status UI
- `animate-in zoom-in-75`, `slide-in-from-top-2` entrance effects
  (`fade-in` may stay)
- The standalone `ShiftWindowBanner` component (its info moves into the
  status bar)

## Acceptance criteria

- `npx tsc --noEmit` passes in `frontend/`.
- `/kiosk` renders: solid header with title + live clock, camera visible
  in the middle, solid status bar at bottom.
- Status bar shows the emerald/sky/neutral scanning variant matching the
  current shift window, turns red on hint, turns bright green with the
  employee name on recognition.
- No element on the page uses backdrop-blur, gradients, or pill-rounded
  corners.
- Recognition, face tracking brackets, TTS, and error overlays all still
  work (behavior untouched).
