"use client";

import * as React from "react";
import {
  CameraOff,
  CheckCircle2,
  Loader2,
  ScanFace,
  WifiOff,
} from "lucide-react";
import { useRecognition } from "@/components/kiosk/use-recognition";

// The kiosk is theme-locked dark regardless of the dashboard's light/dark
// setting, so colors are hardcoded (zinc neutrals + a single emerald accent).

function Clock() {
  // Null until the client ticks — keeps SSR output time-free (no hydration
  // mismatch) and the first tick lands within 1s, behind the init overlay.
  const [now, setNow] = React.useState<Date | null>(null);
  React.useEffect(() => {
    const id = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(id);
  }, []);
  if (!now) return <div className="h-[4.25rem]" aria-hidden />; // reserve space, avoid CLS
  return (
    <div className="text-center">
      <div className="font-mono text-4xl font-semibold tabular-nums tracking-tight text-white sm:text-5xl">
        {now.toLocaleTimeString("vi-VN", {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })}
      </div>
      <div className="mt-1 text-sm text-zinc-400">
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

function CornerFrame({ active }: { active: boolean }) {
  const stroke = active ? "border-emerald-400" : "border-zinc-600";
  const base = "absolute h-14 w-14 transition-colors duration-500";
  return (
    <>
      <div className={`${base} left-0 top-0 rounded-tl-3xl border-l-2 border-t-2 ${stroke}`} />
      <div className={`${base} right-0 top-0 rounded-tr-3xl border-r-2 border-t-2 ${stroke}`} />
      <div className={`${base} bottom-0 left-0 rounded-bl-3xl border-b-2 border-l-2 ${stroke}`} />
      <div className={`${base} bottom-0 right-0 rounded-br-3xl border-b-2 border-r-2 ${stroke}`} />
    </>
  );
}

function Overlay({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center gap-6 bg-zinc-950/85 px-6 text-center backdrop-blur-sm animate-in fade-in duration-300">
      {children}
    </div>
  );
}

export function KioskScreen() {
  const { videoRef, phase, greeting, hint } = useRecognition();
  const scanning = phase === "scanning";

  return (
    <main className="relative min-h-[100dvh] overflow-hidden bg-zinc-950 text-white">
      {/* Live camera fills the screen, mirrored like a selfie view. */}
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
      />
      {/* Cinematic scrim so overlaid text stays legible over any background. */}
      <div className="absolute inset-0 bg-gradient-to-b from-zinc-950/70 via-zinc-950/30 to-zinc-950/80" />
      <div className="pointer-events-none absolute inset-0 shadow-[inset_0_0_180px_60px_rgba(0,0,0,0.7)]" />

      {/* Header: brand + clock */}
      <header className="absolute inset-x-0 top-0 z-10 flex flex-col items-center gap-4 px-6 pt-8">
        <div className="flex items-center gap-2 text-sm font-medium uppercase tracking-[0.2em] text-zinc-300">
          <ScanFace className="h-5 w-5 text-emerald-400" aria-hidden />
          Điểm danh khuôn mặt
        </div>
        <Clock />
      </header>

      {/* Scan reticle */}
      <div className="absolute inset-0 z-10 flex items-center justify-center">
        <div className="relative aspect-[3/4] w-[min(70vw,420px)]">
          <CornerFrame active={scanning} />
          {scanning && (
            <div className="absolute inset-6 rounded-2xl border border-emerald-400/40 animate-pulse motion-reduce:animate-none" />
          )}
        </div>
      </div>

      {/* Scanning hint / prompt */}
      <div
        className="absolute inset-x-0 bottom-0 z-10 flex justify-center px-6 pb-12"
        aria-live="polite"
      >
        <p className="rounded-full bg-zinc-950/60 px-5 py-2 text-base text-zinc-200 backdrop-blur-sm">
          {scanning
            ? (hint ?? "Vui lòng nhìn vào camera để điểm danh")
            : " "}
        </p>
      </div>

      {/* Greeting on a successful recognition (shown ~5s, capture paused) */}
      {phase === "recognized" && greeting && (
        <Overlay>
          <CheckCircle2
            className="h-24 w-24 text-emerald-400 animate-in zoom-in-75 duration-300 motion-reduce:animate-none"
            aria-hidden
          />
          <div>
            <p className="text-lg text-zinc-400">Xin chào</p>
            <p className="mt-1 text-5xl font-semibold text-white sm:text-6xl">
              {greeting.name}
            </p>
          </div>
          <p className="text-2xl font-medium text-emerald-300">
            {greeting.message}
          </p>
        </Overlay>
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
            <p className="text-2xl font-medium text-zinc-200">
              Mất kết nối máy chủ
            </p>
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
