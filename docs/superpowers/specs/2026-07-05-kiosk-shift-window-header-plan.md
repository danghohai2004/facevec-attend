# Plan: move shift-window info to the kiosk header center

**For:** implementing agent (GPT 5.5)
**Scope:** one file — `frontend/src/components/kiosk/kiosk-screen.tsx`. No hooks,
no other files, no behavior change.

## Goal

Today the shift-window status (Giờ vào ca / Giờ tan ca / Ngoài giờ chấm công)
renders in the bottom status bar together with the person-facing prompts. Move
the shift-window info up into the **top header, centered** between the "Chấm
công" poster tag (top-left) and the clock (top-right).

Keep in the bottom bar (unchanged): the greeting ("Xin chào …"), the warning
`hint`, and the come-closer prompt ("Đưa khuôn mặt lại gần hơn").

## Step 1 — Add a `ShiftBadge` component

`AttendanceKind` is already imported in this file. Add this component (e.g. just
above `StatusBar`):

```tsx
/** Ambient shift-window indicator shown centered in the header. Stable (changes
 *  ~1/min), so it lives apart from the live bottom status bar. undefined = clock/
 *  settings not ready yet → render nothing (the grid column collapses, no shift). */
function ShiftBadge({ shiftWindow }: { shiftWindow: AttendanceKind | undefined }) {
  if (shiftWindow === undefined) return null;
  let cls = "bg-card text-muted-foreground";
  let label = "Ngoài giờ chấm công";
  if (shiftWindow === "check_in") {
    cls = "bg-poster-lime text-ink";
    label = "→ Giờ vào ca";
  } else if (shiftWindow === "check_out") {
    cls = "bg-poster-cyan text-ink";
    label = "← Giờ tan ca";
  }
  return (
    <span
      className={`justify-self-center rounded-[3px] border-2 border-foreground px-3 py-1 font-heading text-lg font-black uppercase tracking-tight shadow-brutal-sm sm:text-xl ${cls}`}
    >
      {label}
    </span>
  );
}
```

## Step 2 — Make the header a 3-column grid and drop in the badge

Replace the current header block:

```tsx
      {/* Solid header bar: system identity left (poster tag), clock right. */}
      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between border-b-2 border-foreground bg-background px-8 py-4">
        <p className="rounded-[3px] border-2 border-foreground bg-poster-yellow px-3 py-1 font-heading text-2xl font-black uppercase tracking-tight text-ink shadow-brutal-sm sm:text-3xl">
          Chấm công
        </p>
        <Clock now={now} />
      </header>
```

with:

```tsx
      {/* Header: identity left, shift-window center, clock right. grid-cols
          [1fr_auto_1fr] keeps the center badge truly centered regardless of the
          title/clock widths. */}
      <header className="absolute inset-x-0 top-0 z-20 grid grid-cols-[1fr_auto_1fr] items-center gap-4 border-b-2 border-foreground bg-background px-8 py-4">
        <p className="justify-self-start rounded-[3px] border-2 border-foreground bg-poster-yellow px-3 py-1 font-heading text-2xl font-black uppercase tracking-tight text-ink shadow-brutal-sm sm:text-3xl">
          Chấm công
        </p>
        <ShiftBadge shiftWindow={shiftWindow} />
        <div className="justify-self-end">
          <Clock now={now} />
        </div>
      </header>
```

## Step 3 — Remove the shift-window branches from `StatusBar`

The bottom bar no longer needs `shiftWindow`. In `StatusBar`:

1. Delete the `shiftWindow` field from its props type and destructuring.
2. Replace the trailing branches — the current
   `check_in` / `check_out` / `shiftWindow === undefined` / out-of-hours block:

```tsx
  } else if (phase === "scanning" && shiftWindow === "check_in") {
    barClass = "bg-poster-lime text-ink";
    content = (
      <span>
        <span aria-hidden>→ </span>Giờ vào ca — đưa khuôn mặt vào khung
      </span>
    );
  } else if (phase === "scanning" && shiftWindow === "check_out") {
    barClass = "bg-poster-cyan text-ink";
    content = (
      <span>
        <span aria-hidden>← </span>Giờ tan ca — đưa khuôn mặt vào khung
      </span>
    );
  } else if (phase === "scanning" && shiftWindow === undefined) {
    content = <span>Đưa khuôn mặt vào khung để điểm danh</span>;
  } else if (phase === "scanning") {
    content = <span>Ngoài giờ chấm công</span>;
  }
```

   with a single scanning fallback (default `bg-card text-muted-foreground` bar):

```tsx
  } else if (phase === "scanning") {
    content = <span>Đưa khuôn mặt vào khung để điểm danh</span>;
  }
```

Leave the `recognized` (poster-lime greeting), `hint` (destructive), and `far`
(poster-yellow) branches exactly as they are.

## Step 4 — Update the `<StatusBar>` call site

In `KioskScreen`, remove the `shiftWindow={shiftWindow}` prop from `<StatusBar>`.
Keep the `shiftWindow` derivation (`currentShiftWindow(...)`) — it now feeds
`<ShiftBadge>` in the header. Also remove `shiftWindow` from `StatusBar`'s prop
type (Step 3.1).

## Verification

- `cd frontend && npx tsc --noEmit && npx eslint src/components/kiosk/kiosk-screen.tsx` — must be clean.
- Visual: the centered window badge appears in the header; past a shift boundary
  (or after editing shift-settings) the badge swaps lime→cyan→neutral; the bottom
  bar shows only greeting / warning / come-closer / the idle invite, never the
  window text. When settings/clock aren't loaded yet the badge is absent and the
  title/clock don't move.

## Do NOT

- Touch `use-face-tracker.ts`, `use-recognition.ts`, or `lib/kiosk.ts`.
- Change `kiosk-enrollment.tsx` (registration header has no shift window).
- Add `aria-live` to the header badge — it's ambient, not an announcement; the
  bottom bar keeps its `aria-live="polite"`.
