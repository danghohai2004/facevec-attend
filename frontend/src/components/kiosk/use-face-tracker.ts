"use client";

import * as React from "react";
import { FaceDetector, FilesetResolver } from "@mediapipe/tasks-vision";

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
      raf = requestAnimationFrame(loop);
      const video = videoRef.current;
      const boxEl = boxRef.current;
      if (!detector || !video || !boxEl) return;
      if (video.readyState < 2 || !video.videoWidth) return;
      // detectForVideo requires a monotonically increasing, unique timestamp;
      // skip frames the camera hasn't advanced past.
      if (video.currentTime === lastVideoTime) return;
      lastVideoTime = video.currentTime;

      const result = detector.detectForVideo(video, performance.now());
      const dets = result.detections;
      if (!dets || dets.length === 0) {
        hideBox();
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

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks("/mediapipe/wasm");
        detector = await FaceDetector.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/mediapipe/blaze_face_short_range.tflite",
            delegate: "GPU",
          },
          runningMode: "VIDEO",
        });
        if (cancelled) {
          detector.close();
          detector = null;
          return;
        }
        setStatus("active");
        loop();
      } catch (err) {
        console.error("Face tracker init failed, falling back to server box", err);
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
  }, [videoRef, boxRef]);

  return status;
}
