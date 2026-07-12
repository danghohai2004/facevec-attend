"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  CameraOff,
  CheckCircle2,
  Loader2,
  WifiOff,
} from "lucide-react";
import { useRecognition } from "@/components/kiosk/use-recognition";
import {
  useFaceTracker,
  type Proximity,
} from "@/components/kiosk/use-face-tracker";
import { Brackets, Overlay } from "@/components/kiosk/kiosk-chrome";
import {
  currentShiftWindow,
  type AttendanceKind,
  type FaceBox as FaceBoxCoords,
  type KioskPhase,
  type KioskState,
} from "@/lib/kiosk";
import { getShiftSettings } from "@/lib/api";

// Fullscreen industrial attendance terminal: clear camera feed, a real-time
// bounding box tracking the face (MediaPipe in-browser), high-contrast status.
// Theme-locked dark chrome; the camera image is shown bright (no vignette).

/** Ticks once a second. Shared by the clock and the shift-window status so
 *  there's a single timer, not one per consumer. */
function useNow(): Date | null {
  const [now, setNow] = React.useState<Date | null>(null);
  React.useEffect(() => {
    // Tick each minute-aligned; 1s interval is fine and keeps it simple.
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  return now;
}

function Clock({ now }: { now: Date | null }) {
  if (!now) return <div className="h-12 w-44" aria-hidden />; // reserve space
  return (
    <div className="text-right">
      <div className="font-mono text-3xl font-bold tabular-nums leading-none text-foreground sm:text-4xl">
        {now.toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
        })}
      </div>
      <div className="mt-1 text-sm font-medium uppercase tracking-wide text-muted-foreground sm:text-base">
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

/** Ambient shift-window indicator shown centered in the header. Stable (changes
 *  ~1/min), so it lives apart from the live bottom status bar. undefined = clock/
 *  settings not ready yet → render nothing (the grid column collapses, no shift). */
function ShiftBadge({ shiftWindow }: { shiftWindow: AttendanceKind | undefined }) {
  if (shiftWindow === undefined) return null;
  let cls = "bg-card text-foreground";
  let label: React.ReactNode = "Ngoài giờ chấm công";
  if (shiftWindow === "check_in") {
    cls = "bg-poster-lime text-ink";
    label = <><span aria-hidden>→ </span>Giờ vào ca</>;
  } else if (shiftWindow === "check_out") {
    cls = "bg-poster-cyan text-ink";
    label = <><span aria-hidden>← </span>Giờ tan ca</>;
  }
  return (
    <span
      className={`justify-self-center rounded-[3px] border-2 border-foreground px-3 py-1 font-heading text-lg font-black uppercase tracking-tight shadow-brutal-sm sm:text-xl ${cls}`}
    >
      {label}
    </span>
  );
}

/** One solid floating panel for live, person-facing status at the bottom.
 *  Priority: recognition result > warning hint > proximity hint.
 *  Solid colors, no blur, uppercase — readable across a room.
 *
 *  When none of those priority states applies, render nothing so the aria-live
 *  region doesn't announce text that contradicts the screen state. */
function StatusBar({
  phase,
  greeting,
  hint,
  proximity,
}: {
  phase: KioskPhase;
  greeting: KioskState["greeting"];
  hint: string | null;
  proximity: Proximity;
}) {
  let barClass = "bg-card text-muted-foreground";
  let content: React.ReactNode = null;

  if (phase === "recognized" && greeting) {
    barClass = "bg-poster-lime text-ink";
    content = (
      <>
        <CheckCircle2 className="h-8 w-8 shrink-0 sm:h-10 sm:w-10" aria-hidden />
        <span>
          Xin chào {greeting.name} — {greeting.message}
        </span>
      </>
    );
  } else if (phase === "scanning" && hint) {
    barClass = "bg-destructive text-white";
    content = (
      <>
        <AlertTriangle className="h-8 w-8 shrink-0 sm:h-10 sm:w-10" aria-hidden />
        <span>{hint}</span>
      </>
    );
  } else if (phase === "scanning" && proximity === "far") {
    barClass = "bg-poster-yellow text-ink";
    content = (
      <>
        <AlertTriangle className="h-8 w-8 shrink-0 sm:h-10 sm:w-10" aria-hidden />
        <span>Đưa khuôn mặt lại gần hơn</span>
      </>
    );
  }

  if (!content) return null;

  return (
    <div
      className={`absolute bottom-4 left-1/2 z-20 flex min-h-20 max-w-[calc(100%-2rem)] w-max -translate-x-1/2 items-center justify-center gap-4 rounded-lg border-2 border-foreground px-8 py-4 text-center font-heading text-xl font-black uppercase tracking-tight shadow-brutal sm:text-3xl ${barClass}`}
      aria-live="polite"
    >
      {content}
    </div>
  );
}

/**
 * Fallback box (used only if the in-browser tracker fails to init): draws the
 * backend model bbox. Same object-cover + mirror math, but updates ~1×/sec.
 */
function ServerFaceBox({
  videoRef,
  bbox,
}: {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  bbox: FaceBoxCoords;
}) {
  const [rect, setRect] = React.useState<{
    left: number;
    top: number;
    width: number;
    height: number;
  } | null>(null);

  React.useEffect(() => {
    let raf = 0;
    const compute = () => {
      const el = videoRef.current;
      if (!el) return setRect(null);
      const cw = el.clientWidth;
      const ch = el.clientHeight;
      const fw = el.videoWidth;
      const fh = el.videoHeight;
      if (!fw || !fh || !cw || !ch) return setRect(null);
      const scale = Math.max(cw / fw, ch / fh);
      const offX = (cw - fw * scale) / 2;
      const offY = (ch - fh * scale) / 2;
      const [nx1, ny1, nx2, ny2] = bbox;
      const px1 = offX + nx1 * fw * scale;
      const px2 = offX + nx2 * fw * scale;
      const py1 = offY + ny1 * fh * scale;
      const py2 = offY + ny2 * fh * scale;
      setRect({ left: cw - px2, top: py1, width: px2 - px1, height: py2 - py1 });
    };
    const schedule = () => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(compute);
    };
    schedule();
    window.addEventListener("resize", schedule);
    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", schedule);
    };
  }, [videoRef, bbox]);

  if (!rect) return null;
  const corner = Math.max(16, Math.min(44, Math.min(rect.width, rect.height) * 0.22));
  return (
    <div
      className="pointer-events-none absolute z-10 transition-all duration-200 ease-out motion-reduce:transition-none"
      style={{ left: rect.left, top: rect.top, width: rect.width, height: rect.height }}
      aria-hidden
    >
      <Brackets size={corner} />
    </div>
  );
}

export function KioskScreen() {
  // True once a tracked face is close enough to mean it, not just passing
  // through the shot. Written by useFaceTracker, read by useRecognition to
  // gate capture without per-frame re-renders; only proximity transitions
  // update React state for the status bar.
  const canCaptureRef = React.useRef(true);
  const { videoRef, phase, greeting, hint, faceBox } = useRecognition(canCaptureRef);
  const boxRef = React.useRef<HTMLDivElement>(null);
  const { status: trackerStatus, proximity } = useFaceTracker(
    videoRef,
    boxRef,
    canCaptureRef,
  );
  const showServerBox = trackerStatus === "failed" && faceBox && phase !== "recognized";

  const now = useNow();
  // The kiosk tab runs for hours without remounting or losing focus, so a
  // plain useQuery only ever fetches once at boot — an admin editing shift
  // settings elsewhere would never reach it without refetchInterval polling.
  const shiftQuery = useQuery({
    queryKey: ["shift-settings"],
    queryFn: getShiftSettings,
    staleTime: 60 * 1000,
    refetchInterval: 60 * 1000,
  });
  // undefined = clock/settings not ready yet — ShiftBadge stays absent until
  // the window is actually known.
  const shiftWindow =
    now && shiftQuery.data ? currentShiftWindow(now, shiftQuery.data) : undefined;

  return (
    <main className="dark relative min-h-[100dvh] overflow-hidden bg-background text-foreground">
      {/* Live camera fills the screen, shown bright and mirrored (selfie view). */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
      />

      {/* Real-time in-browser face box (positioned imperatively by the tracker). */}
      <div ref={boxRef} className="pointer-events-none absolute z-10 hidden" aria-hidden>
        <Brackets />
      </div>

      {/* Fallback box from the backend model if the in-browser tracker failed. */}
      {showServerBox && <ServerFaceBox videoRef={videoRef} bbox={faceBox} />}

      {/* Header: identity left, shift-window center, clock right. grid-cols
          [1fr_auto_1fr] keeps the center badge truly centered regardless of the
          title/clock widths. */}
      <header className="absolute inset-x-4 top-4 z-20 grid grid-cols-[1fr_auto_1fr] items-center gap-4 rounded-lg border-2 border-foreground bg-background px-6 py-3 shadow-brutal">
        <p className="justify-self-start rounded-[3px] border-2 border-foreground bg-poster-yellow px-3 py-1 font-heading text-2xl font-black uppercase tracking-tight text-ink shadow-brutal-sm sm:text-3xl">
          Chấm công
        </p>
        <ShiftBadge shiftWindow={shiftWindow} />
        <div className="justify-self-end">
          <Clock now={now} />
        </div>
      </header>

      <StatusBar
        phase={phase}
        greeting={greeting}
        hint={hint}
        proximity={proximity}
      />

      {/* Camera permission / hardware failure */}
      {phase === "camera_error" && (
        <Overlay>
          <CameraOff className="h-20 w-20 text-poster-yellow" aria-hidden />
          <div>
            <p className="font-heading text-3xl font-black uppercase tracking-tight">
              Không truy cập được camera
            </p>
            <p className="mt-3 text-base text-muted-foreground">
              Vui lòng cấp quyền camera cho trình duyệt và tải lại trang.
            </p>
          </div>
        </Overlay>
      )}

      {/* Booting the camera */}
      {phase === "initializing" && (
        <Overlay>
          <Loader2 className="h-16 w-16 animate-spin text-poster-lime motion-reduce:animate-none" aria-hidden />
          <p className="font-heading text-2xl font-black uppercase tracking-tight text-foreground">
            Đang khởi động camera…
          </p>
        </Overlay>
      )}

      {/* Socket dropped, auto-reconnecting */}
      {phase === "disconnected" && (
        <Overlay>
          <WifiOff className="h-16 w-16 text-poster-yellow" aria-hidden />
          <div>
            <p className="font-heading text-2xl font-black uppercase tracking-tight text-foreground">
              Mất kết nối máy chủ
            </p>
            <p className="mt-2 flex items-center justify-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden />
              Đang kết nối lại…
            </p>
          </div>
        </Overlay>
      )}
    </main>
  );
}
