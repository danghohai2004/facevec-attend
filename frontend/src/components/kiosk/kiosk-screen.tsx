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
import { useFaceTracker } from "@/components/kiosk/use-face-tracker";
import {
  currentShiftWindow,
  type AttendanceKind,
  type FaceBox as FaceBoxCoords,
  type KioskPhase,
} from "@/lib/kiosk";
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

      <StatusBar
        phase={phase}
        greeting={greeting}
        hint={hint}
        shiftWindow={shiftWindow}
      />

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
