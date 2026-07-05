// Runnable self-check for the kiosk reducer (no test runner in this project):
//   node --experimental-strip-types src/lib/kiosk.test.ts
import assert from "node:assert/strict";
import {
  initialKioskState,
  reduceKiosk,
  kioskPhase,
  shouldCapture,
  recognitionWsUrl,
  attendanceKind,
  currentShiftWindow,
  enrollmentUrl,
  isFaceCloseEnough,
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

// An unknown result carries the model bbox → the face box is drawn.
s = reduceKiosk(s, {
  type: "message",
  message: { status: "unknown", bbox: [0.3, 0.2, 0.6, 0.7], timestamp: "t" },
});
assert.deepEqual(s.faceBox, [0.3, 0.2, 0.6, 0.7]);
// no_face clears the box.
assert.equal(reduceKiosk(s, { type: "message", message: { status: "no_face", timestamp: "t" } }).faceBox, null);
// A spoof result is silent and clears the box so normal shift guidance can show.
const spoofed = reduceKiosk(s, {
  type: "message",
  message: { status: "spoof", timestamp: "t" },
});
assert.equal(spoofed.hint, null);
assert.equal(spoofed.faceBox, null);

// A recognition greeting pauses capture and shows Vietnamese copy.
s = reduceKiosk(s, {
  type: "message",
  message: {
    status: "recognized",
    emp_id: 1,
    name: "Trần Minh",
    attendance: "Check in successful",
    bbox: [0.3, 0.2, 0.6, 0.7],
    timestamp: "t",
  },
});
assert.equal(kioskPhase(s), "recognized");
assert.equal(shouldCapture(s), false);
assert.deepEqual(s.greeting, {
  name: "Trần Minh",
  message: "Đã chấm công vào ca",
  kind: "check_in",
});

// In-flight results are ignored while the greeting is up (no flicker).
const held = reduceKiosk(s, {
  type: "message",
  message: { status: "unknown", bbox: [0, 0, 1, 1], timestamp: "t" },
});
assert.equal(held.greeting?.name, "Trần Minh");

// Greeting clears → back to scanning, box cleared.
s = reduceKiosk(s, { type: "greeting_done" });
assert.equal(kioskPhase(s), "scanning");
assert.equal(s.faceBox, null);

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

// Raw backend strings map to the right badge kind for the greeting card.
assert.equal(attendanceKind("Check in successful"), "check_in");
assert.equal(attendanceKind("Already checked in"), "check_in");
assert.equal(attendanceKind("Check out successful"), "check_out");
assert.equal(attendanceKind("Check in not found to check out"), "check_out");
assert.equal(attendanceKind("Not during working hours"), null);
assert.equal(attendanceKind("Lỗi hệ thống"), null);

// currentShiftWindow mirrors the backend's window check, including overnight wrap.
const shift = {
  checkInStart: "08:00",
  checkInEnd: "10:00",
  checkOutStart: "17:00",
  checkOutEnd: "19:00",
};
assert.equal(currentShiftWindow(new Date(2026, 0, 1, 9, 0), shift), "check_in");
assert.equal(currentShiftWindow(new Date(2026, 0, 1, 18, 0), shift), "check_out");
assert.equal(currentShiftWindow(new Date(2026, 0, 1, 12, 0), shift), null);
// Overnight window (22:00 → 02:00) wraps past midnight.
const overnight = { ...shift, checkOutStart: "22:00", checkOutEnd: "02:00" };
assert.equal(currentShiftWindow(new Date(2026, 0, 1, 23, 30), overnight), "check_out");
assert.equal(currentShiftWindow(new Date(2026, 0, 1, 1, 30), overnight), "check_out");

// isFaceCloseEnough gates capture on bbox area vs. frame area, not raw pixels.
const frame = { w: 1280, h: 720 };
assert.equal(isFaceCloseEnough({ width: 100, height: 100 }, frame.w, frame.h), false); // far away, passing by
assert.equal(isFaceCloseEnough({ width: 400, height: 350 }, frame.w, frame.h), true); // close enough
assert.equal(isFaceCloseEnough({ width: 400, height: 350 }, 0, 0), false); // no frame dims yet

// enrollmentUrl: query string must survive spaces, unicode, and separators.
{
  const url = enrollmentUrl("Nguyễn A&B", "EMP 1004");
  const qs = new URLSearchParams(url.split("?")[1]);
  assert.equal(url.startsWith("/kiosk?mode=register"), true);
  assert.equal(qs.get("mode"), "register");
  assert.equal(qs.get("name"), "Nguyễn A&B");
  assert.equal(qs.get("emp_code"), "EMP 1004");
}

console.log("kiosk.test.ts: all assertions passed");
