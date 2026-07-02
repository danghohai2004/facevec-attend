// Runnable self-check for the kiosk reducer (no test runner in this project):
//   node --experimental-strip-types src/lib/kiosk.test.ts
import assert from "node:assert/strict";
import {
  initialKioskState,
  reduceKiosk,
  kioskPhase,
  shouldCapture,
  recognitionWsUrl,
  type KioskState,
} from "./kiosk.ts";

// Fresh boot: camera pending, socket connecting → initializing, don't capture.
assert.equal(kioskPhase(initialKioskState), "initializing");
assert.equal(shouldCapture(initialKioskState), false);

// Camera + socket both up → scanning, and now we capture.
let s: KioskState = initialKioskState;
s = reduceKiosk(s, { type: "camera_ready" });
s = reduceKiosk(s, { type: "ws_open" });
assert.equal(kioskPhase(s), "scanning");
assert.equal(shouldCapture(s), true);

// A recognition greeting pauses capture and shows Vietnamese copy.
s = reduceKiosk(s, {
  type: "message",
  message: {
    status: "recognized",
    emp_id: 1,
    name: "Trần Minh",
    attendance: "Check in successful",
    timestamp: "t",
  },
});
assert.equal(kioskPhase(s), "recognized");
assert.equal(shouldCapture(s), false);
assert.deepEqual(s.greeting, { name: "Trần Minh", message: "Đã chấm công vào ca" });

// In-flight results are ignored while the greeting is up (no flicker).
const held = reduceKiosk(s, { type: "message", message: { status: "unknown", timestamp: "t" } });
assert.equal(held.greeting?.name, "Trần Minh");

// Greeting clears → back to scanning.
s = reduceKiosk(s, { type: "greeting_done" });
assert.equal(kioskPhase(s), "scanning");

// Camera error dominates every other state.
assert.equal(kioskPhase(reduceKiosk(s, { type: "camera_error" })), "camera_error");

// Socket drop while camera is fine → disconnected.
assert.equal(kioskPhase(reduceKiosk(s, { type: "ws_close" })), "disconnected");

// URL scheme flips http→ws / https→wss.
assert.equal(
  recognitionWsUrl("http://localhost:8000", "kiosk-1"),
  "ws://localhost:8000/ws/recognition/kiosk-1",
);
assert.equal(
  recognitionWsUrl("https://api.example.com/", "a b"),
  "wss://api.example.com/ws/recognition/a%20b",
);

console.log("kiosk.test.ts: all assertions passed");
