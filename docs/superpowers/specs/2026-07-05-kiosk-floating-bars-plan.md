# Plan: floating rounded header/footer + drop the idle invite

**For:** implementing agent (GPT 5.5)
**Scope:** two files —
`frontend/src/components/kiosk/kiosk-screen.tsx` and
`frontend/src/components/kiosk/kiosk-enrollment.tsx`. No hooks/logic changes.

## Goal

1. Remove the idle status text "Đưa khuôn mặt vào khung để điểm danh" — when the
   kiosk is scanning with nothing to say, show **no** bottom bar.
2. Header and footer should be **floating rounded panels** inset from the screen
   edges (margins on all sides + full border + hard shadow), not full-width black
   bars that touch the left/right edges. The camera feed shows through around
   them.

Keep the current content: header = "Chấm công" tag / shift badge / clock; footer
= greeting, warning `hint`, come-closer. Same poster status colors.

## kiosk-screen.tsx

### 1. Drop the idle branch and hide the empty bar in `StatusBar`

Delete this branch entirely (the last `else if`):

```tsx
  } else if (phase === "scanning") {
    content = <span>Đưa khuôn mặt vào khung để điểm danh</span>;
  }
```

Then, right before the `return`, bail out when there is nothing to show so no
empty panel floats at the bottom:

```tsx
  if (!content) return null;

  return (
    <div ...>
```

Now the footer only appears for recognized / hint / far; scanning-idle and every
non-scanning phase render nothing (overlays already cover those).

### 2. Float the footer panel

Replace the `StatusBar` container className:

```tsx
      className={`absolute inset-x-0 bottom-0 z-20 flex min-h-20 items-center justify-center gap-4 border-t-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight sm:text-3xl ${barClass}`}
```

with (inset from edges, full border, rounded, hard shadow):

```tsx
      className={`absolute inset-x-4 bottom-4 z-20 flex min-h-20 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
```

### 3. Float the header panel

Replace the header opening tag:

```tsx
      <header className="absolute inset-x-0 top-0 z-20 grid grid-cols-[1fr_auto_1fr] items-center gap-4 border-b-2 border-foreground bg-background px-8 py-4">
```

with:

```tsx
      <header className="absolute inset-x-4 top-4 z-20 grid grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-lg border-2 border-foreground bg-background px-6 py-3 shadow-brutal">
```

(Inner content — title tag, `<ShiftBadge>`, clock — unchanged.)

## kiosk-enrollment.tsx

Apply the same float treatment so the register screen stays consistent. Its
footer always has text (keep its default "…để đăng ký" — do **not** remove it),
so no null-guard is needed there.

### Header — replace:

```tsx
      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between border-b-2 border-foreground bg-background px-8 py-4">
```

with:

```tsx
      <header className="absolute inset-x-4 top-4 z-20 flex items-center justify-between gap-4 rounded-lg border-2 border-foreground bg-background px-6 py-3 shadow-brutal">
```

### Footer — replace:

```tsx
        className={`absolute inset-x-0 bottom-0 z-20 flex min-h-20 items-center justify-center gap-4 border-t-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight sm:text-3xl ${barClass}`}
```

with:

```tsx
        className={`absolute inset-x-4 bottom-4 z-20 flex min-h-20 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
```

## Notes / rationale

- `rounded-lg` = the design system's `--radius` (~10px) — modest, on-brand with
  the brutalist cards, not a pill.
- `shadow-brutal` (white hard offset, since the kiosk is dark-locked) gives the
  floated panels the poster "lifted" look over the camera feed.
- Full `border-2` all around (was `border-b-2`/`border-t-2`) closes the frame now
  that the panels don't touch the screen edges.

## Verification

- `cd frontend && npx tsc --noEmit && npx eslint src/components/kiosk/kiosk-screen.tsx src/components/kiosk/kiosk-enrollment.tsx` — clean.
- Visual: header/footer are rounded cards with a gap to all four edges and the
  camera visible around them. With a face far → yellow footer card; recognized →
  lime card; scanning-idle → **no footer at all**. Enrollment keeps its footer.

## Do NOT

- Touch `use-face-tracker.ts`, `use-recognition.ts`, `lib/kiosk.ts`, or the
  overlays/brackets.
- Remove the enrollment idle text — only the attendance screen drops its idle
  invite.
