# Kiosk Industrial Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the kiosk attendance screen from cinematic/sci-fi (floating pills, blur, gradients) to an industrial time-clock terminal look (solid opaque bars, flat, square, color-coded states).

**Architecture:** Render-only change confined to `frontend/src/components/kiosk/kiosk-screen.tsx`. The fullscreen mirrored camera stays; the floating chrome is replaced by a solid header bar (title + clock) and one solid full-width bottom status bar that consolidates shift window, guidance, hints, and the recognition greeting. All hooks, WebSocket flow, face tracking, and TTS are untouched.

**Tech Stack:** Next.js (App Router), React 19, Tailwind CSS v4, lucide-react icons. Spec: `docs/superpowers/specs/2026-07-03-kiosk-industrial-redesign-design.md`.

## Global Constraints

- Modify ONLY `frontend/src/components/kiosk/kiosk-screen.tsx`. Do not touch `use-recognition.ts`, `use-face-tracker.ts`, `frontend/src/lib/kiosk.ts`, or anything under `src/` (Python backend).
- This Next.js version may differ from your training data — read `frontend/node_modules/next/dist/docs/` if unsure about an API (per `frontend/AGENTS.md`).
- No new dependencies.
- Final page must contain no `backdrop-blur-*`, no `bg-gradient-*`, no `rounded-full`/`rounded-2xl` status pills, no `zoom-in`/`slide-in` entrance animations (`fade-in` is allowed).
- UI copy is Vietnamese — copy the exact strings from this plan (they contain diacritics; do not retype them).
- All commands below run from `frontend/` unless stated otherwise.
- Verification commands: `npx tsc --noEmit` (typecheck) and `curl -s http://localhost:3000/kiosk | grep -o "<string>"` (dev server renders — the dev server is usually already running on :3000).

---

### Task 1: Restyle face brackets (square, white)

**Files:**
- Modify: `frontend/src/components/kiosk/kiosk-screen.tsx:98-110` (the `Brackets` component)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: `Brackets({ size?: number })` — same signature, only classes change. Used unchanged by the tracker box div and `ServerFaceBox`.

- [ ] **Step 1: Replace the `Brackets` component body**

Current code:

```tsx
/** draw_bbox-style corner brackets. Rendered inside a positioned box. */
function Brackets({ size = 28 }: { size?: number }) {
  const c = "absolute border-amber-300";
  const s = { width: size, height: size };
  return (
    <>
      <span className={`${c} left-0 top-0 rounded-tl-md border-l-[3px] border-t-[3px]`} style={s} />
      <span className={`${c} right-0 top-0 rounded-tr-md border-r-[3px] border-t-[3px]`} style={s} />
      <span className={`${c} bottom-0 left-0 rounded-bl-md border-b-[3px] border-l-[3px]`} style={s} />
      <span className={`${c} bottom-0 right-0 rounded-br-md border-b-[3px] border-r-[3px]`} style={s} />
    </>
  );
}
```

Replace with (white, no rounded corners):

```tsx
/** Corner brackets around the tracked face. Square and white — industrial,
 *  not HUD. Rendered inside a positioned box. */
function Brackets({ size = 28 }: { size?: number }) {
  const c = "absolute border-white";
  const s = { width: size, height: size };
  return (
    <>
      <span className={`${c} left-0 top-0 border-l-[3px] border-t-[3px]`} style={s} />
      <span className={`${c} right-0 top-0 border-r-[3px] border-t-[3px]`} style={s} />
      <span className={`${c} bottom-0 left-0 border-b-[3px] border-l-[3px]`} style={s} />
      <span className={`${c} bottom-0 right-0 border-b-[3px] border-r-[3px]`} style={s} />
    </>
  );
}
```

- [ ] **Step 2: Typecheck**

Run: `npx tsc --noEmit`
Expected: exits 0, no output.

- [ ] **Step 3: Commit**

```bash
git add src/components/kiosk/kiosk-screen.tsx
git commit -m "style(kiosk): square white face brackets"
```

---

### Task 2: Solid header bar (title + clock), remove gradients

**Files:**
- Modify: `frontend/src/components/kiosk/kiosk-screen.tsx` (the `Clock` component, the two gradient divs, and the `<header>` element)

**Interfaces:**
- Consumes: existing `useNow()` hook (unchanged) and `Clock({ now })` component (classes change, signature unchanged).
- Produces: a `<header>` bar that Task 3's status bar mirrors in style (`bg-zinc-950`, opaque, bordered).

- [ ] **Step 1: Restyle `Clock`**

Current code:

```tsx
function Clock({ now }: { now: Date | null }) {
  if (!now) return <div className="h-14 w-44" aria-hidden />; // reserve space
  return (
    <div className="text-right">
      <div className="font-mono text-2xl font-semibold tabular-nums leading-none text-white sm:text-3xl">
        {now.toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </div>
      <div className="mt-2 text-base font-medium text-zinc-200 sm:text-lg">
        {now.toLocaleDateString("vi-VN", {
          weekday: "long",
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
        })}
      </div>
    </div>
  );
}
```

Replace with (same data, tighter industrial type):

```tsx
function Clock({ now }: { now: Date | null }) {
  if (!now) return <div className="h-12 w-44" aria-hidden />; // reserve space
  return (
    <div className="text-right">
      <div className="font-mono text-3xl font-bold tabular-nums leading-none text-white sm:text-4xl">
        {now.toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </div>
      <div className="mt-1 text-sm font-medium uppercase tracking-wide text-zinc-400 sm:text-base">
        {now.toLocaleDateString("vi-VN", {
          weekday: "long",
          day: "2-digit",
          month: "2-digit",
          year: "numeric",
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Delete the two gradient divs**

Delete these two lines (directly after the `<video>` element in `KioskScreen`):

```tsx
      {/* Light gradients only where text sits — keep the face itself clear. */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-40 bg-gradient-to-b from-zinc-950/85 to-transparent" />
      <div className="pointer-events-none absolute inset-x-0 bottom-0 h-40 bg-gradient-to-t from-zinc-950/90 to-transparent" />
```

- [ ] **Step 3: Replace the floating header with a solid bar**

Current code:

```tsx
      {/* Clock, top-right */}
      <header className="absolute inset-x-0 top-0 z-20 flex items-start justify-end px-8 pt-6">
        <Clock now={now} />
      </header>
```

Replace with:

```tsx
      {/* Solid header bar: system identity left, clock right. */}
      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between border-b-2 border-zinc-800 bg-zinc-950 px-8 py-4">
        <div>
          <p className="text-2xl font-black uppercase tracking-tight text-white sm:text-3xl">
            Chấm công
          </p>
          <p className="text-sm font-medium text-zinc-400 sm:text-base">
            Hệ thống điểm danh khuôn mặt
          </p>
        </div>
        <Clock now={now} />
      </header>
```

- [ ] **Step 4: Typecheck and render check**

Run: `npx tsc --noEmit`
Expected: exits 0.

Run: `curl -s http://localhost:3000/kiosk | grep -c "Hệ thống điểm danh khuôn mặt"`
Expected: `1` (or more). If the dev server is not running, start it: `npm run dev` (background), wait for ready, then curl.

- [ ] **Step 5: Commit**

```bash
git add src/components/kiosk/kiosk-screen.tsx
git commit -m "style(kiosk): solid industrial header bar, drop gradients"
```

---

### Task 3: Consolidated solid status bar (replaces pills, banner, greeting circle)

**Files:**
- Modify: `frontend/src/components/kiosk/kiosk-screen.tsx` (delete `shiftKindCopy` + `ShiftWindowBanner`; delete the scanning/hint/greeting JSX blocks; add `StatusBar` component and mount it)

**Interfaces:**
- Consumes: `phase: KioskPhase`, `greeting: { name: string; message: string; kind: AttendanceKind } | null`, `hint: string | null` from the existing `useRecognition` return value; `shiftWindow: AttendanceKind` from the existing `currentShiftWindow(now, shiftQuery.data)` call. `AttendanceKind` is `"check_in" | "check_out" | null`.
- Produces: `StatusBar({ phase, greeting, hint, shiftWindow })` mounted at the bottom of `KioskScreen`.

- [ ] **Step 1: Update the lib import to include `KioskPhase`**

Current import in `kiosk-screen.tsx`:

```tsx
import { currentShiftWindow, type AttendanceKind, type FaceBox as FaceBoxCoords } from "@/lib/kiosk";
```

Replace with:

```tsx
import {
  currentShiftWindow,
  type AttendanceKind,
  type FaceBox as FaceBoxCoords,
  type KioskPhase,
} from "@/lib/kiosk";
```

(`KioskPhase` is already exported from `frontend/src/lib/kiosk.ts` — do not edit that file.)

- [ ] **Step 2: Delete `shiftKindCopy` and `ShiftWindowBanner`**

Delete this entire block (the `Record` const with `bannerClass` values and the banner component, roughly lines 57–96 of the current file):

```tsx
// Tailwind can't see interpolated class names ("text-${accent}-300"), so each
// variant's full class strings are spelled out here instead of built at runtime.
const shiftKindCopy: Record<
  ...
> = { ... };

/** Big pill, top-center, telling the person which side of the shift the
 *  terminal is currently in. ... */
function ShiftWindowBanner({ kind }: { kind: AttendanceKind }) {
  ...
}
```

Also remove `LogIn` and `LogOut` from the `lucide-react` import (they are only used by the deleted code):

```tsx
import {
  AlertTriangle,
  CameraOff,
  CheckCircle2,
  Loader2,
  WifiOff,
} from "lucide-react";
```

- [ ] **Step 3: Add the `StatusBar` component**

Add this component where `ShiftWindowBanner` used to be:

```tsx
/** One solid full-width bar at the bottom — the single place all status goes.
 *  Priority: recognition result > warning hint > shift-window guidance >
 *  out-of-hours. Solid colors, no blur, uppercase — readable across a room. */
function StatusBar({
  phase,
  greeting,
  hint,
  shiftWindow,
}: {
  phase: KioskPhase;
  greeting: { name: string; message: string; kind: AttendanceKind } | null;
  hint: string | null;
  shiftWindow: AttendanceKind;
}) {
  let barClass = "bg-zinc-900 text-zinc-400";
  let content: React.ReactNode = "Ngoài giờ chấm công";

  if (phase === "recognized" && greeting) {
    barClass = "bg-green-600 text-white";
    content = (
      <>
        <CheckCircle2 className="h-8 w-8 shrink-0 sm:h-10 sm:w-10" aria-hidden />
        <span>
          Xin chào {greeting.name} — {greeting.message}
        </span>
      </>
    );
  } else if (phase === "scanning" && hint) {
    barClass = "bg-red-700 text-white";
    content = (
      <>
        <AlertTriangle className="h-8 w-8 shrink-0 sm:h-10 sm:w-10" aria-hidden />
        <span>{hint}</span>
      </>
    );
  } else if (phase === "scanning" && shiftWindow === "check_in") {
    barClass = "bg-emerald-700 text-white";
    content = <span>→ Giờ vào ca — đưa khuôn mặt vào khung</span>;
  } else if (phase === "scanning" && shiftWindow === "check_out") {
    barClass = "bg-sky-700 text-white";
    content = <span>← Giờ tan ca — đưa khuôn mặt vào khung</span>;
  }

  return (
    <div
      className={`absolute inset-x-0 bottom-0 z-20 flex min-h-20 items-center justify-center gap-4 border-t-2 border-zinc-800 px-8 py-4 text-center text-xl font-black uppercase tracking-tight sm:text-3xl ${barClass}`}
      aria-live="polite"
    >
      {content}
    </div>
  );
}
```

- [ ] **Step 4: Replace the three floating status blocks with the bar**

In `KioskScreen`'s JSX, delete all three of these blocks:

1. The shift banner mount:

```tsx
      {/* Shift-window status, top-center. ... */}
      {shiftWindow && (
        <div className="pointer-events-none absolute inset-x-0 top-6 z-20 flex justify-center px-6">
          <ShiftWindowBanner kind={shiftWindow} />
        </div>
      )}
```

2. The bottom scanning/hint block (`{scanning && ( ... )}` — the one containing the red hint pill and the "Đưa khuôn mặt vào giữa màn hình để điểm danh" pill).

3. The greeting checkmark circle block (`{phase === "recognized" && greeting && ( ... )}` — the one with the `sr-only` span). The `StatusBar`'s visible text plus `aria-live` replaces the sr-only pattern; TTS in `use-recognition.ts` is untouched and still speaks.

In their place (just before the `{/* Camera permission / hardware failure */}` overlay block), add:

```tsx
      <StatusBar
        phase={phase}
        greeting={greeting}
        hint={hint}
        shiftWindow={shiftWindow}
      />
```

The `scanning` const (`const scanning = phase === "scanning";`) becomes unused — delete it.

- [ ] **Step 5: Typecheck and render check**

Run: `npx tsc --noEmit`
Expected: exits 0. If it reports unused imports/variables, delete exactly those leftovers.

Run: `curl -s http://localhost:3000/kiosk | grep -c "border-t-2"`
Expected: `1` or more.

- [ ] **Step 6: Commit**

```bash
git add src/components/kiosk/kiosk-screen.tsx
git commit -m "style(kiosk): consolidate status into one solid color-coded bar"
```

---

### Task 4: Flatten full-screen overlays + acceptance sweep

**Files:**
- Modify: `frontend/src/components/kiosk/kiosk-screen.tsx` (the `Overlay` component and the greeting `animate-in` remnants)

**Interfaces:**
- Consumes: `Overlay({ children })` — signature unchanged.
- Produces: final page satisfying the spec's acceptance criteria.

- [ ] **Step 1: Flatten `Overlay`**

Current code:

```tsx
function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-6 bg-zinc-950/90 px-6 text-center animate-in fade-in duration-300">
      {children}
    </div>
  );
}
```

Replace with (fully opaque, fade only):

```tsx
function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-6 bg-zinc-950 px-6 text-center animate-in fade-in duration-300">
      {children}
    </div>
  );
}
```

- [ ] **Step 2: Acceptance sweep — no leftover cinematic styling**

Run from `frontend/`:

```bash
grep -nE "backdrop-blur|bg-gradient|rounded-full|rounded-2xl|zoom-in|slide-in" src/components/kiosk/kiosk-screen.tsx
```

Expected: no output. If anything matches, remove that styling (per the Global Constraints) and re-run.

- [ ] **Step 3: Full verification**

Run: `npx tsc --noEmit`
Expected: exits 0.

Run: `npm run lint 2>&1 | tail -5` (if the project has no lint script, skip)
Expected: no errors in `kiosk-screen.tsx`.

Render check: `curl -s http://localhost:3000/kiosk | grep -c "Chấm công"`
Expected: `1` or more.

Manual check (report to the user, do not skip): open `http://localhost:3000/kiosk` in a browser —
- solid header with title left, live clock right;
- camera visible and mirrored in the middle;
- solid bottom bar showing the emerald "GIỜ VÀO CA", sky "GIỜ TAN CA", or neutral "NGOÀI GIỜ CHẤM CÔNG" variant depending on current time vs. shift settings;
- white square brackets track the face;
- on a successful recognition the bar turns bright green with the employee's name and TTS still speaks.

- [ ] **Step 4: Commit**

```bash
git add src/components/kiosk/kiosk-screen.tsx
git commit -m "style(kiosk): flatten overlays, finish industrial restyle"
```
