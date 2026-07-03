"use client";

import * as React from "react";
import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  CameraOff,
  CheckCircle2,
  Loader2,
  LogIn,
  LogOut,
  WifiOff,
} from "lucide-react";
import { useRecognition } from "@/components/kiosk/use-recognition";
import { useFaceTracker } from "@/components/kiosk/use-face-tracker";
import { currentShiftWindow, type AttendanceKind, type FaceBox as FaceBoxCoords } from "@/lib/kiosk";
import { getShiftSettings } from "@/lib/api";

// Fullscreen industrial attendance terminal: clear camera feed, a real-time
// bounding box tracking the face (MediaPipe in-browser), high-contrast status.
// Theme-locked dark chrome; the camera image is shown bright (no vignette).

/** Ticks once a second. Shared by the clock and the shift-window badge so
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

// Tailwind can't see interpolated class names ("text-${accent}-300"), so each
// variant's full class strings are spelled out here instead of built at runtime.
const shiftKindCopy: Record<
  NonNullable<AttendanceKind>,
  {
    label: string;
    icon: typeof LogIn;
    bannerClass: string;
  }
> = {
  check_in: {
    label: "Giờ vào ca",
    icon: LogIn,
    bannerClass: "bg-emerald-600 shadow-[0_4px_24px_-4px] shadow-emerald-950/60",
  },
  check_out: {
    label: "Giờ tan ca",
    icon: LogOut,
    bannerClass: "bg-sky-600 shadow-[0_4px_24px_-4px] shadow-sky-950/60",
  },
};

/** Big pill, top-center, telling the person which side of the shift the
 *  terminal is currently in. Independent of scan/recognition phase so it sits
 *  still and keeps ticking over in real time (only hidden behind the
 *  full-screen camera/connection overlays, which sit at a higher z-index). */
function ShiftWindowBanner({ kind }: { kind: AttendanceKind }) {
  if (!kind) return null;
  const { label, icon: Icon, bannerClass } = shiftKindCopy[kind];
  return (
    <div
      className={`flex items-center gap-3 rounded-full px-6 py-3 sm:gap-4 sm:px-9 sm:py-4 animate-in fade-in slide-in-from-top-2 duration-300 ${bannerClass}`}
    >
      <Icon className="h-7 w-7 shrink-0 text-white sm:h-9 sm:w-9" aria-hidden />
      <p className="text-xl font-black uppercase tracking-tight text-white sm:text-3xl">
        {label}
      </p>
    </div>
  );
}

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

function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-6 bg-zinc-950/90 px-6 text-center animate-in fade-in duration-300">
      {children}
    </div>
  );
}

export function KioskScreen() {
  // True once a tracked face is close enough to mean it, not just passing
  // through the shot. Written by useFaceTracker, read by useRecognition to
  // gate capture — a plain ref so proximity changes don't cause re-renders.
  const canCaptureRef = React.useRef(true);
  const { videoRef, phase, greeting, hint, faceBox } = useRecognition(canCaptureRef);
  const boxRef = React.useRef<HTMLDivElement>(null);
  const trackerStatus = useFaceTracker(videoRef, boxRef, canCaptureRef);
  const scanning = phase === "scanning";
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
  const shiftWindow =
    now && shiftQuery.data ? currentShiftWindow(now, shiftQuery.data) : null;

  return (
    <main className="relative min-h-[100dvh] overflow-hidden bg-zinc-950 text-white">
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

      {/* Shift-window status, top-center. Stays up and keeps updating in real
          time across scanning AND the post-recognition greeting — it is not
          gated on `phase`, only on knowing a window at all. The full-screen
          camera/connection overlays below sit at z-30 and cover it naturally
          when there's nothing useful to show anyway. */}
      {shiftWindow && (
        <div className="pointer-events-none absolute inset-x-0 top-6 z-20 flex justify-center px-6">
          <ShiftWindowBanner kind={shiftWindow} />
        </div>
      )}

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

      {/* Bottom status bar. A hint means something needs attention → small red
          warning pill; otherwise the big neutral guidance prompt. */}
      {scanning && (
        <div
          className="absolute inset-x-0 bottom-0 z-20 flex justify-center px-6 pb-10"
          aria-live="polite"
        >
          {hint ? (
            <div className="flex items-center gap-2 rounded-lg bg-red-950/70 px-4 py-2 backdrop-blur-md ring-1 ring-red-500/40">
              <AlertTriangle className="h-4 w-4 shrink-0 text-red-400" aria-hidden />
              <p className="text-sm font-medium text-red-100 sm:text-base">{hint}</p>
            </div>
          ) : (
            <div className="flex items-center gap-3 rounded-2xl bg-zinc-950/70 px-7 py-4 backdrop-blur-md ring-1 ring-white/10">
              <span
                className="h-2.5 w-2.5 rounded-full bg-emerald-400 motion-safe:animate-pulse"
                aria-hidden
              />
              <p className="text-xl font-medium text-white sm:text-2xl">
                Đưa khuôn mặt vào giữa màn hình để điểm danh
              </p>
            </div>
          )}
        </div>
      )}

      {/* Greeting on a successful recognition (~5s, capture paused). Icon-only
          now — the name/status text is spoken aloud instead (see
          useRecognition's speech announcement) so the camera view stays
          uncluttered. The sr-only text keeps the same info reachable for
          screen readers, since audio-only excludes anyone who can't hear it. */}
      {phase === "recognized" && greeting && (
        <div
          className="absolute inset-x-0 bottom-0 z-30 flex justify-center pb-10"
          aria-live="polite"
        >
          <div className="flex h-24 w-24 items-center justify-center rounded-full bg-zinc-950/70 backdrop-blur-md ring-1 ring-emerald-400/30 animate-in zoom-in-75 fade-in duration-300 motion-reduce:animate-none">
            <CheckCircle2 className="h-14 w-14 text-emerald-400" aria-hidden />
          </div>
          <span className="sr-only">
            Xin chào {greeting.name}. {greeting.message}
          </span>
        </div>
      )}

      {/* Camera permission / hardware failure */}
      {phase === "camera_error" && (
        <Overlay>
          <CameraOff className="h-20 w-20 text-amber-400" aria-hidden />
          <div className="max-w-md">
            <p className="text-3xl font-semibold">Không truy cập được camera</p>
            <p className="mt-3 text-base text-zinc-400">
              Vui lòng cấp quyền camera cho trình duyệt và tải lại trang.
            </p>
          </div>
        </Overlay>
      )}

      {/* Booting the camera */}
      {phase === "initializing" && (
        <Overlay>
          <Loader2 className="h-16 w-16 animate-spin text-emerald-400 motion-reduce:animate-none" aria-hidden />
          <p className="text-2xl font-medium text-zinc-200">Đang khởi động camera…</p>
        </Overlay>
      )}

      {/* Socket dropped, auto-reconnecting */}
      {phase === "disconnected" && (
        <Overlay>
          <WifiOff className="h-16 w-16 text-amber-400" aria-hidden />
          <div>
            <p className="text-2xl font-medium text-zinc-200">Mất kết nối máy chủ</p>
            <p className="mt-2 flex items-center justify-center gap-2 text-sm text-zinc-400">
              <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden />
              Đang kết nối lại…
            </p>
          </div>
        </Overlay>
      )}
    </main>
  );
}
