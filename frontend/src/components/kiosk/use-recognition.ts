"use client";

import * as React from "react";
import {
  attendanceMessage,
  initialKioskState,
  kioskPhase,
  recognitionWsUrl,
  reduceKiosk,
  shouldCapture,
  type RecognitionMessage,
} from "@/lib/kiosk";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
const CAPTURE_INTERVAL_MS = 1000;
const GREETING_MS = 5000;
const RECONNECT_MS = 2000;
const JPEG_QUALITY = 0.7;

// The greeting card shows an icon only (no name/status text — see
// kiosk-screen.tsx), so this is now the only way that info reaches the
// person: read it aloud. Synthesized server-side (src/modules/tts) rather
// than the browser's Web Speech API — installed voices vary wildly by
// OS/browser (silent on machines with no Vietnamese voice, robotic on
// others), so every kiosk sounds the same regardless of what's on it.
async function speakGreeting(name: string, attendance: string) {
  const text = `Xin chào ${name}. ${attendanceMessage(attendance)}.`;
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/tts?text=${encodeURIComponent(text)}`,
    );
    if (!response.ok) return;
    const url = URL.createObjectURL(await response.blob());
    const audio = new Audio(url);
    audio.addEventListener("ended", () => URL.revokeObjectURL(url));
    audio.addEventListener("error", () => URL.revokeObjectURL(url));
    await audio.play().catch(() => {});
  } catch {
    // Best-effort — a failed announcement shouldn't break the kiosk flow.
  }
}

// One kiosk terminal keeps a stable id so backend logs stay attributable.
// ?client_id= wins (per-device provisioning), else a persisted random id.
function resolveClientId(): string {
  const fromUrl = new URLSearchParams(window.location.search).get("client_id");
  if (fromUrl) return fromUrl;
  const KEY = "kiosk-client-id";
  let id = window.localStorage.getItem(KEY);
  if (!id) {
    id = crypto.randomUUID();
    window.localStorage.setItem(KEY, id);
  }
  return id;
}

export function useRecognition(canCaptureRef?: React.RefObject<boolean>) {
  const videoRef = React.useRef<HTMLVideoElement>(null);
  const [state, dispatch] = React.useReducer(reduceKiosk, initialKioskState);

  // Long-lived callbacks (capture interval, socket handlers) must see the
  // latest state without re-subscribing, so mirror it into a ref. The ref is
  // synced in an effect (never during render); async readers run well after commit.
  const stateRef = React.useRef(state);
  React.useEffect(() => {
    stateRef.current = state;
  }, [state]);

  const wsRef = React.useRef<WebSocket | null>(null);
  const greetingTimerRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    let stream: MediaStream | null = null;
    let reconnectTimer: number | null = null;
    let captureTimer: number | null = null;

    const clientId = resolveClientId();
    const wsUrl = recognitionWsUrl(API_BASE_URL, clientId);
    const canvas = document.createElement("canvas");

    function connect() {
      if (cancelled) return;
      dispatch({ type: "ws_connecting" });
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;
      ws.onopen = () => dispatch({ type: "ws_open" });
      ws.onmessage = (ev) => {
        let msg: RecognitionMessage;
        try {
          msg = JSON.parse(ev.data as string);
        } catch {
          return;
        }
        dispatch({ type: "message", message: msg });
        if (msg.status === "recognized") {
          void speakGreeting(msg.name, msg.attendance);
          if (greetingTimerRef.current) {
            window.clearTimeout(greetingTimerRef.current);
          }
          greetingTimerRef.current = window.setTimeout(
            () => dispatch({ type: "greeting_done" }),
            GREETING_MS,
          );
        }
      };
      ws.onclose = () => {
        wsRef.current = null;
        dispatch({ type: "ws_close" });
        if (!cancelled) {
          reconnectTimer = window.setTimeout(connect, RECONNECT_MS);
        }
      };
      ws.onerror = () => ws.close();
    }

    function capture() {
      const video = videoRef.current;
      const ws = wsRef.current;
      if (!video || !video.videoWidth) return;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (!shouldCapture(stateRef.current)) return;
      // Skip the frame if the in-browser tracker says nobody's close enough to
      // mean it (someone merely walking past). Left unset/true when the
      // tracker isn't actively running, so this never blocks capture on
      // machines where in-browser tracking failed to start.
      if (canCaptureRef && canCaptureRef.current === false) return;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      canvas.toBlob(
        (blob) => {
          if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(blob);
          }
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    }

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user", width: 1280, height: 720 },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play().catch(() => {});
        }
        dispatch({ type: "camera_ready" });
        connect();
        captureTimer = window.setInterval(capture, CAPTURE_INTERVAL_MS);
      } catch {
        dispatch({ type: "camera_error" });
      }
    }

    start();

    return () => {
      cancelled = true;
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
      if (captureTimer) window.clearInterval(captureTimer);
      if (greetingTimerRef.current) window.clearTimeout(greetingTimerRef.current);
      wsRef.current?.close();
      wsRef.current = null;
      stream?.getTracks().forEach((t) => t.stop());
    };
  }, [canCaptureRef]);

  return {
    videoRef,
    phase: kioskPhase(state),
    greeting: state.greeting,
    hint: state.hint,
    faceBox: state.faceBox,
  };
}
