# Plan: stop the footer greeting from wrapping (px-10 → px-8)

**For:** implementing agent (GPT 5.5)
**Scope:** two files —
`frontend/src/components/kiosk/kiosk-screen.tsx` (`StatusBar`) and
`frontend/src/components/kiosk/kiosk-enrollment.tsx` (footer `<div>`).
One class change per file. No logic/hooks.

## Problem

A long greeting like "Xin chào Đặng Hồ Hải — Ngoài khung giờ chấm công" now wraps
to two lines. Root cause: the previous hug change bumped padding to `px-10`. The
panel width is capped at `max-w-[calc(100%-2rem)]` (same width the old
`inset-x-4` bar had), so text room = `max-w − horizontal padding`. `px-10` (2.5rem
each side) leaves ~1rem less room than the old `inset-x-4 px-8` layout, which was
just enough to tip a borderline-long greeting into wrapping.

## Fix

Revert the padding `px-10` → `px-8` in both footer classNames. This restores the
exact one-line text room the old full-width `inset-x-4 px-8` bar had (both use the
same `100%-2rem` width cap), so long greetings fit on one line again — while the
hug behavior still shrinks the panel for short messages. `px-8` (2rem/side) stays
comfortably roomy.

### kiosk-screen.tsx — `StatusBar` container
Change `... rounded-lg border-2 border-foreground px-10 py-4 ...`
to `... rounded-lg border-2 border-foreground px-8 py-4 ...`
(only `px-10` → `px-8`; everything else identical).

### kiosk-enrollment.tsx — footer `<div>`
Same single change: `px-10` → `px-8`.

## Verification

- `cd frontend && npx tsc --noEmit && npx eslint src/components/kiosk/kiosk-screen.tsx src/components/kiosk/kiosk-enrollment.tsx` — clean.
- Visual: the greeting "Xin chào Đặng Hồ Hải — Ngoài khung giờ chấm công" sits on
  one line; short states ("Đưa khuôn mặt lại gần hơn") still hug into a compact
  centered card. (A genuinely longer-than-screen greeting may still wrap — that's
  the intended safe fallback, not a regression.)

## Do NOT

- Add `whitespace-nowrap` (would overflow the screen for very long names).
- Touch the header, overlays, brackets, hooks, or `max-w`/centering classes.
