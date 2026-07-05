"use client";

import * as React from "react";

const JPEG_QUALITY = 0.85;

/** Grab the current video frame as a JPEG File. Returns null if the video
 *  isn't producing pixels yet (no dimensions). The backend enrollment endpoint
 *  consumes a single representative image, so one frame is all we send. */
export function captureFrame(video: HTMLVideoElement): File | null {
  if (!video.videoWidth || !video.videoHeight) return null;

  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const context = canvas.getContext("2d");
  if (!context) return null;

  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataUrl = canvas.toDataURL("image/jpeg", JPEG_QUALITY);
  const base64 = dataUrl.split(",")[1] ?? "";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }

  return new File([bytes], `enroll-${Date.now()}.jpg`, { type: "image/jpeg" });
}

/** Camera lifecycle for the enrollment screen: open getUserMedia into videoRef,
 *  report boot/ready/error. No WebSocket, no capture loop — the screen decides
 *  when to capture.
 *  ponytail: duplicates useRecognition's camera-start (~15 lines). Upgrade path:
 *  extract a shared useCameraStream(videoRef) if a third camera consumer appears. */
export function useEnrollmentCamera(
  videoRef: React.RefObject<HTMLVideoElement | null>,
): { phase: "initializing" | "ready" | "error" } {
  const [phase, setPhase] = React.useState<"initializing" | "ready" | "error">(
    "initializing",
  );

  React.useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 1280, height: 720 },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((track) => track.stop());
          return;
        }

        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play().catch(() => {});
        }
        setPhase("ready");
      } catch {
        if (!cancelled) setPhase("error");
      }
    }

    start();
    return () => {
      cancelled = true;
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [videoRef]);

  return { phase };
}
