"use client";

import * as React from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";
import { isFaceCloseEnough } from "@/lib/kiosk";

// Real-time, in-browser face box. MediaPipe BlazeFace runs on every animation
// frame and positions `boxRef` imperatively (no React re-render per frame, so it
// stays smooth). The backend still owns identity + attendance; this is only the
// live box the user sees tracking their face.
//
// Assets are self-hosted under /public/mediapipe so the kiosk works offline.

export type TrackerStatus = "loading" | "active" | "failed";

export function useFaceTracker(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  boxRef: React.RefObject<HTMLDivElement | null>,
  // Mutated imperatively (not React state) so a face merely walking past
  // doesn't cause a render on every frame. `useRecognition` reads this ref
  // to decide whether a capture is even worth sending. While the tracker
  // isn't actively running (loading at boot, or permanently failed on a
  // machine with no WebGL), this is left `true` so capture behaves exactly
  // as it did before the proximity gate existed — no in-browser detection
  // means no cheap way to gate, so we fall back to sending every tick.
  canCaptureRef?: React.RefObject<boolean>,
): TrackerStatus {
  const [status, setStatus] = React.useState<TrackerStatus>("loading");

  React.useEffect(() => {
    let detector: FaceDetector | null = null;
    let raf = 0;
    let cancelled = false;
    let lastVideoTime = -1;

    function hideBox() {
      const el = boxRef.current;
      if (el) el.style.display = "none";
    }

    function loop() {
      const video = videoRef.current;
      const boxEl = boxRef.current;
      if (!detector || !video || !boxEl) {
        raf = requestAnimationFrame(loop);
        return;
      }
      if (video.readyState < 2 || !video.videoWidth) {
        raf = requestAnimationFrame(loop);
        return;
      }
      // detectForVideo requires a monotonically increasing, unique timestamp;
      // skip frames the camera hasn't advanced past.
      if (video.currentTime === lastVideoTime) {
        raf = requestAnimationFrame(loop);
        return;
      }
      lastVideoTime = video.currentTime;

      let result;
      try {
        result = detector.detectForVideo(video, performance.now());
      } catch (err) {
        // Some environments report a usable delegate at creation time but
        // fail on the first real inference (e.g. no WebGL context at all —
        // MediaPipe's Web build needs GL for buffer plumbing regardless of
        // delegate). Stop tracking and let the server-box fallback take over
        // instead of leaving an uncaught error in the RAF loop.
        console.error("Face tracker inference failed, falling back to server box", err);
        hideBox();
        if (canCaptureRef) canCaptureRef.current = true;
        setStatus("failed");
        return;
      }
      raf = requestAnimationFrame(loop);
      const dets = result.detections;
      if (!dets || dets.length === 0) {
        hideBox();
        if (canCaptureRef) canCaptureRef.current = false;
        return;
      }
      // Largest detection = nearest camera, matching the backend's face pick.
      let best = dets[0];
      for (const d of dets) {
        const a = d.boundingBox!.width * d.boundingBox!.height;
        const b = best.boundingBox!.width * best.boundingBox!.height;
        if (a > b) best = d;
      }
      const bb = best.boundingBox!; // pixels in the video's intrinsic frame

      const fw = video.videoWidth;
      const fh = video.videoHeight;
      if (canCaptureRef) canCaptureRef.current = isFaceCloseEnough(bb, fw, fh);
      const cw = video.clientWidth;
      const ch = video.clientHeight;
      // object-cover: frame scaled to cover, centered, cropped.
      const scale = Math.max(cw / fw, ch / fh);
      const offX = (cw - fw * scale) / 2;
      const offY = (ch - fh * scale) / 2;
      const px1 = offX + bb.originX * scale;
      const px2 = offX + (bb.originX + bb.width) * scale;
      const py1 = offY + bb.originY * scale;
      const py2 = offY + (bb.originY + bb.height) * scale;
      // Video is mirrored (-scale-x-100), so flip x around the container.
      boxEl.style.display = "block";
      boxEl.style.left = `${cw - px2}px`;
      boxEl.style.top = `${py1}px`;
      boxEl.style.width = `${px2 - px1}px`;
      boxEl.style.height = `${py2 - py1}px`;
    }

    async function createDetector(vision: Awaited<ReturnType<typeof FilesetResolver.forVisionTasks>>) {
      const baseOptions = { modelAssetPath: "/mediapipe/blaze_face_short_range.tflite" };
      try {
        // GPU is faster, but needs a working WebGL context — not guaranteed on
        // every machine/browser (e.g. no WebGL support, software rendering
        // disabled). Fall back to CPU rather than losing in-browser tracking
        // entirely over a delegate that just isn't available here.
        return await FaceDetector.createFromOptions(vision, {
          baseOptions: { ...baseOptions, delegate: "GPU" },
          runningMode: "VIDEO",
        });
      } catch (err) {
        console.warn("GPU face detector unavailable, retrying on CPU", err);
        return await FaceDetector.createFromOptions(vision, {
          baseOptions: { ...baseOptions, delegate: "CPU" },
          runningMode: "VIDEO",
        });
      }
    }

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        detector = await createDetector(vision);
        if (cancelled) {
          detector.close();
          detector = null;
          return;
        }
        setStatus("active");
        loop();
      } catch (err) {
        console.error("Face tracker init failed, falling back to server box", err);
        if (canCaptureRef) canCaptureRef.current = true;
        setStatus("failed");
      }
    }

    init();

    return () => {
      cancelled = true;
      cancelAnimationFrame(raf);
      detector?.close();
      detector = null;
    };
  }, [videoRef, boxRef, canCaptureRef]);

  return status;
}
