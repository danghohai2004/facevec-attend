# Plan: footer panel hugs its text instead of spanning full width

**For:** implementing agent (GPT 5.5)
**Scope:** two files —
`frontend/src/components/kiosk/kiosk-screen.tsx` and
`frontend/src/components/kiosk/kiosk-enrollment.tsx`. No hooks/logic changes.
**Header is intentionally NOT changed** (it holds left/center/right items and
should keep spanning).

## Problem

The bottom status panel uses `inset-x-4`, so it stretches nearly edge-to-edge
with the text centered and large empty colored margins on both sides (see
`kioss.png`). It makes the camera area feel cramped.

## Goal

The footer panel should size to its content — centered horizontally, with
comfortable (not tight) padding — while never overflowing the screen on long
text like a full greeting.

## kiosk-screen.tsx — `StatusBar` container

Replace the container className:

```tsx
      className={`absolute inset-x-4 bottom-4 z-20 flex min-h-20 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
```

with:

```tsx
      className={`absolute bottom-4 left-1/2 z-20 flex min-h-20 max-w-[calc(100%-2rem)] -translate-x-1/2 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-10 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
```

What changed and why:
- `inset-x-4` removed → the panel no longer stretches full width; it shrinks to
  fit its content.
- `left-1/2 -translate-x-1/2` → centers the now content-sized panel horizontally.
- `max-w-[calc(100%-2rem)]` → caps width so a long greeting wraps inside the panel
  and still keeps a ~1rem gap to each screen edge, never overflowing.
- `px-8` → `px-10` → slightly roomier side padding so the text isn't cramped
  against the border ("bo sát hơn nhưng không gò bó"). `min-h-20`/`py-4` keep the
  vertical breathing room.

## kiosk-enrollment.tsx — footer `<div>`

Apply the identical change for consistency. Replace:

```tsx
        className={`absolute inset-x-4 bottom-4 z-20 flex min-h-20 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
```

with:

```tsx
        className={`absolute bottom-4 left-1/2 z-20 flex min-h-20 max-w-[calc(100%-2rem)] -translate-x-1/2 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-10 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
```

## Verification

- `cd frontend && npx tsc --noEmit && npx eslint src/components/kiosk/kiosk-screen.tsx src/components/kiosk/kiosk-enrollment.tsx` — clean.
- Visual: short states ("Đưa khuôn mặt lại gần hơn") sit in a compact centered
  pill-card; a full greeting grows the panel but stops ~1rem short of each edge
  and wraps inside if needed. Header unchanged.

## Do NOT

- Change the header, overlays, brackets, or any hook/logic.
- Reintroduce a removed idle message.
