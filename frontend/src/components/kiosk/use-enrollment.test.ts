// Runnable self-check for enrollment frame capture (no test runner in this project):
//   node --experimental-strip-types src/components/kiosk/use-enrollment.test.ts
import assert from "node:assert/strict";
import { captureFrame } from "./use-enrollment.ts";

// A video that is not producing pixels yet cannot be captured.
assert.equal(
  captureFrame({ videoWidth: 0, videoHeight: 720 } as HTMLVideoElement),
  null,
);

// A live frame is drawn at its intrinsic dimensions and returned as a JPEG File.
let drawImageArgs: unknown[] | null = null;
const canvas = {
  width: 0,
  height: 0,
  getContext: () => ({
    drawImage: (...args: unknown[]) => {
      drawImageArgs = args;
    },
  }),
  toDataURL: () => "data:image/jpeg;base64,aGVsbG8=",
};

Object.defineProperty(globalThis, "document", {
  configurable: true,
  value: {
    createElement: (tagName: string) => {
      assert.equal(tagName, "canvas");
      return canvas;
    },
  },
});

const video = { videoWidth: 1280, videoHeight: 720 } as HTMLVideoElement;
const file = captureFrame(video);
assert.ok(file instanceof File);
assert.equal(file.type, "image/jpeg");
assert.match(file.name, /^enroll-\d+\.jpg$/);
assert.equal(canvas.width, 1280);
assert.equal(canvas.height, 720);
assert.deepEqual(drawImageArgs, [video, 0, 0, 1280, 720]);

console.log("use-enrollment.test.ts: all assertions passed");
