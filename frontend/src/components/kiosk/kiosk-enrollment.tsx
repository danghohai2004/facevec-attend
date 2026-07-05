"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { CameraOff, CheckCircle2, Loader2, UserPlus } from "lucide-react";
import { Brackets, Overlay } from "@/components/kiosk/kiosk-chrome";
import {
  captureFrame,
  ENROLLMENT_BURST_COUNT,
  ENROLLMENT_BURST_INTERVAL_MS,
  nextEnrollmentCountdown,
  useEnrollmentCamera,
} from "@/components/kiosk/use-enrollment";
import { useFaceTracker } from "@/components/kiosk/use-face-tracker";
import { createEmployee } from "@/lib/api";

const SUCCESS_REDIRECT_MS = 3000;
const ERROR_RESET_MS = 3000;

function MissingEmployeeInfo() {
  const router = useRouter();

  return (
    <main className="relative min-h-[100dvh] overflow-hidden bg-zinc-950 text-white">
      <Overlay>
        <UserPlus className="h-20 w-20 text-amber-400" aria-hidden />
        <div className="max-w-md">
          <p className="text-3xl font-semibold">Thiếu thông tin nhân viên</p>
          <p className="mt-3 text-base text-zinc-400">
            Vui lòng quay lại trang Employees và nhập tên + mã nhân viên.
          </p>
          <button
            type="button"
            onClick={() => router.push("/employees")}
            className="mt-6 rounded-[3px] border-2 border-white px-6 py-2 text-base font-bold uppercase"
          >
            Quay lại Employees
          </button>
        </div>
      </Overlay>
    </main>
  );
}

function EnrollmentCapture({
  name,
  empCode,
}: {
  name: string;
  empCode: string;
}) {
  const router = useRouter();
  const queryClient = useQueryClient();
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const boxRef = React.useRef<HTMLDivElement>(null);
  const canCaptureRef = React.useRef(true);
  const submittedRef = React.useRef(false);

  const { phase: cameraPhase } = useEnrollmentCamera(videoRef);
  const { proximity } = useFaceTracker(videoRef, boxRef, canCaptureRef);
  const [countdown, setCountdown] = React.useState<number | null>(null);
  const [capturing, setCapturing] = React.useState(false);

  const {
    error,
    isError,
    isPending,
    isSuccess,
    mutate,
    reset,
  } = useMutation({
    mutationFn: createEmployee,
  });

  React.useEffect(() => {
    if (!isSuccess) return;
    void queryClient.invalidateQueries({ queryKey: ["employees"] });
    void queryClient.invalidateQueries({ queryKey: ["employee-name"] });
    const timeout = window.setTimeout(
      () => router.push("/employees"),
      SUCCESS_REDIRECT_MS,
    );
    return () => window.clearTimeout(timeout);
  }, [isSuccess, queryClient, router]);

  React.useEffect(() => {
    if (!isError) return;
    const timeout = window.setTimeout(() => {
      submittedRef.current = false;
      setCapturing(false);
      setCountdown(null);
      reset();
    }, ERROR_RESET_MS);
    return () => window.clearTimeout(timeout);
  }, [isError, reset]);

  const idle =
    cameraPhase === "ready" &&
    !isError &&
    !isPending &&
    !isSuccess;

  React.useEffect(() => {
    if (!idle || submittedRef.current || proximity !== "ok") return;

    let interval: number | undefined;
    const animationFrame = window.requestAnimationFrame(() => {
      // Updater form: a brief proximity blip resumes the count instead of
      // restarting at 3 (nextEnrollmentCountdown keeps a non-null current).
      setCountdown((current) => nextEnrollmentCountdown(current, "face_ok"));
      interval = window.setInterval(() => {
        setCountdown((current) => nextEnrollmentCountdown(current, "tick"));
      }, 1000);
    });
    return () => {
      window.cancelAnimationFrame(animationFrame);
      if (interval !== undefined) window.clearInterval(interval);
    };
  }, [idle, proximity]);

  React.useEffect(() => {
    if (
      countdown === null ||
      countdown > 0 ||
      proximity !== "ok" ||
      submittedRef.current
    ) {
      return;
    }

    const video = videoRef.current;
    if (!video) return;
    const first = captureFrame(video);
    if (!first) {
      // No pixels yet — retry the countdown instead of submitting a blank frame.
      const animationFrame = window.requestAnimationFrame(() => {
        setCountdown(nextEnrollmentCountdown(null, "face_ok"));
      });
      return () => window.cancelAnimationFrame(animationFrame);
    }

    // Lock in, then collect a short burst. Count only usable (non-null) frames
    // toward the target, but cap total attempts so a face lost mid-burst still
    // submits what we have (>=1, since `first` is already in).
    submittedRef.current = true;
    setCapturing(true);
    const frames: File[] = [first];
    let attempts = 1;
    let cancelled = false;
    const interval = window.setInterval(() => {
      const current = videoRef.current;
      const frame = current ? captureFrame(current) : null;
      if (frame) frames.push(frame);
      attempts += 1;
      if (
        frames.length >= ENROLLMENT_BURST_COUNT ||
        attempts >= ENROLLMENT_BURST_COUNT * 2
      ) {
        window.clearInterval(interval);
        if (!cancelled) mutate({ name, empCode, files: frames });
      }
    }, ENROLLMENT_BURST_INTERVAL_MS);

    return () => {
      cancelled = true;
      window.clearInterval(interval);
    };
  }, [countdown, empCode, mutate, name, proximity]);

  const errorMessage =
    error instanceof Error && error.message
      ? error.message
      : "Không thể đăng ký nhân viên.";

  const statusText = (() => {
    if (isPending) return "Đang đăng ký…";
    if (isError) return `Đăng ký thất bại — ${errorMessage}`;
    if (capturing) return "Đang chụp…";
    if (proximity === "ok" && countdown !== null && countdown > 0) {
      return `Giữ yên… ${countdown}`;
    }
    if (proximity === "far") return "Đưa khuôn mặt lại gần hơn";
    return "Đưa khuôn mặt vào khung để đăng ký";
  })();

  const barClass = isError
    ? "bg-red-700 text-white"
    : capturing || (proximity === "ok" && countdown !== null && countdown > 0)
      ? "bg-emerald-700 text-white"
      : proximity === "far"
        ? "bg-amber-600 text-white"
        : "bg-zinc-900 text-zinc-400";

  return (
    <main className="relative min-h-[100dvh] overflow-hidden bg-zinc-950 text-white">
      <video
        ref={videoRef}
        autoPlay
        muted
        playsInline
        className="absolute inset-0 h-full w-full -scale-x-100 object-cover"
      />

      <div ref={boxRef} className="pointer-events-none absolute z-10 hidden" aria-hidden>
        <Brackets />
      </div>

      <header className="absolute inset-x-0 top-0 z-20 flex items-center justify-between border-b-2 border-zinc-800 bg-zinc-950 px-8 py-4">
        <p className="text-2xl font-black uppercase tracking-tight text-white sm:text-3xl">
          Đăng ký
        </p>
        <p className="text-right text-sm font-medium uppercase tracking-wide text-zinc-400 sm:text-base">
          {name} · {empCode}
        </p>
      </header>

      <div
        className={`absolute inset-x-0 bottom-0 z-20 flex min-h-20 items-center justify-center gap-4 border-t-2 border-zinc-800 px-8 py-4 text-center text-xl font-black uppercase tracking-tight sm:text-3xl ${barClass}`}
        aria-live="polite"
      >
        <span>{statusText}</span>
      </div>

      {cameraPhase === "error" && (
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

      {cameraPhase === "initializing" && (
        <Overlay>
          <Loader2
            className="h-16 w-16 animate-spin text-emerald-400 motion-reduce:animate-none"
            aria-hidden
          />
          <p className="text-2xl font-medium text-zinc-200">
            Đang khởi động camera…
          </p>
        </Overlay>
      )}

      {isSuccess && (
        <Overlay>
          <CheckCircle2 className="h-20 w-20 text-green-400" aria-hidden />
          <p className="text-3xl font-semibold">Đăng ký thành công</p>
          <p className="text-base text-zinc-400">
            Đang quay lại danh sách nhân viên…
          </p>
        </Overlay>
      )}
    </main>
  );
}

export function KioskEnrollment({
  name,
  empCode,
}: {
  name: string;
  empCode: string;
}) {
  if (!name.trim() || !empCode.trim()) {
    return <MissingEmployeeInfo />;
  }

  return <EnrollmentCapture name={name} empCode={empCode} />;
}
