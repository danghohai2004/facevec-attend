// Pure kiosk logic: WebSocket message contract, view state machine, and the
// Vietnamese copy shown to the person standing at the camera. No DOM here so it
// stays trivially testable — see kiosk.test.ts for the reducer self-check.

/** Normalized face box [x1, y1, x2, y2] in 0..1, from the model (un-mirrored). */
export type FaceBox = [number, number, number, number];

/** Messages the backend pushes over /ws/recognition/{client_id} (see pipeline.py). */
export type RecognitionMessage =
  | {
      status: "recognized";
      emp_id: number;
      name: string;
      attendance: string;
      bbox: FaceBox;
      timestamp: string;
    }
  | { status: "unknown"; bbox: FaceBox; timestamp: string }
  | { status: "no_face"; timestamp: string }
  | { status: "spoof"; timestamp: string }
  | { status: "error"; detail: string };

/**
 * Two async resources drive the screen independently: the camera and the
 * socket. Track each rather than flattening early — the derived `kioskPhase`
 * picks the one that matters for what the user sees.
 */
/** Which side of a shift the just-logged (or currently open) event belongs to. */
export type AttendanceKind = "check_in" | "check_out" | null;

export interface KioskState {
  camera: "pending" | "ready" | "error";
  socket: "connecting" | "open" | "closed";
  greeting: { name: string; message: string; kind: AttendanceKind } | null;
  hint: string | null;
  // Live model bbox to draw over the detected face, or null when no face.
  faceBox: FaceBox | null;
}

export type KioskPhase =
  | "initializing"
  | "scanning"
  | "recognized"
  | "camera_error"
  | "disconnected";

export type KioskEvent =
  | { type: "camera_ready" }
  | { type: "camera_error" }
  | { type: "ws_connecting" }
  | { type: "ws_open" }
  | { type: "ws_close" }
  | { type: "message"; message: RecognitionMessage }
  | { type: "greeting_done" };

export const initialKioskState: KioskState = {
  camera: "pending",
  socket: "connecting",
  greeting: null,
  hint: null,
  faceBox: null,
};

/** Backend attendance strings (log_attendance) → friendly Vietnamese kiosk copy. */
export function attendanceMessage(raw: string): string {
  switch (raw) {
    case "Check in successful":
      return "Đã chấm công vào ca";
    case "Check out successful":
      return "Đã chấm công tan ca";
    case "Already checked in":
      return "Bạn đã chấm công vào ca hôm nay";
    case "Check in not found to check out":
      return "Chưa có ca vào để chấm tan ca";
    case "Not during working hours":
      return "Ngoài khung giờ chấm công";
    default:
      // "Lỗi hệ thống" and any future string fall through unchanged.
      return raw;
  }
}

/** Which shift-side a raw backend attendance string belongs to (for the result badge). */
export function attendanceKind(raw: string): AttendanceKind {
  switch (raw) {
    case "Check in successful":
    case "Already checked in":
      return "check_in";
    case "Check out successful":
    case "Check in not found to check out":
      return "check_out";
    default:
      return null;
  }
}

/** "HH:mm" → minutes since midnight. */
function toMinutes(hhmm: string): number {
  const [h, m] = hhmm.split(":").map(Number);
  return h * 60 + m;
}

/** Same wraparound rule as the backend's _is_time_in_range (service.py). */
function inRange(current: number, start: number, end: number): boolean {
  if (start <= end) return current >= start && current <= end;
  return current >= start || current <= end; // overnight shift
}

// Someone merely walking past the camera shouldn't trigger recognition/
// attendance — only a face filling at least this fraction of the frame area
// counts as "close enough to mean it". Tune by testing at the camera's actual
// mount distance; there's no principled derivation, it's a physical heuristic.
export const MIN_FACE_AREA_RATIO = 0.12;
// Once close, keep treating the face as close until it shrinks past this lower
// ratio. Without this band a face held right at MIN_FACE_AREA_RATIO flickers
// "ok"↔"far" frame-to-frame (MediaPipe runs per frame), which restarts the
// enrollment countdown forever. Hysteresis makes the decision stable.
export const KEEP_FACE_AREA_RATIO = 0.1;

/** Whether a missing detection is still within the brief dropout window. */
export function isWithinFaceLossGrace(
  lastSeen: number,
  now: number,
  graceMs: number,
): boolean {
  return now - lastSeen < graceMs;
}

/** Is a detected face's bounding box big enough (as a fraction of the video
 *  frame) to treat as someone actually approaching the kiosk, not just
 *  passing through the shot? `wasClose` applies hysteresis: once close, the
 *  face stays "close" until it drops below the lower keep-threshold. */
export function isFaceCloseEnough(
  bbox: { width: number; height: number },
  frameWidth: number,
  frameHeight: number,
  wasClose = false,
): boolean {
  if (!frameWidth || !frameHeight) return false;
  const ratio = (bbox.width * bbox.height) / (frameWidth * frameHeight);
  return ratio >= (wasClose ? KEEP_FACE_AREA_RATIO : MIN_FACE_AREA_RATIO);
}

/** Which shift window "now" falls in, so the kiosk can hint check-in vs check-out
 *  before the person even scans. Mirrors get_current_time in service.py. */
export function currentShiftWindow(
  now: Date,
  shift: {
    checkInStart: string;
    checkInEnd: string;
    checkOutStart: string;
    checkOutEnd: string;
  },
): AttendanceKind {
  const current = now.getHours() * 60 + now.getMinutes();
  if (inRange(current, toMinutes(shift.checkInStart), toMinutes(shift.checkInEnd))) {
    return "check_in";
  }
  if (inRange(current, toMinutes(shift.checkOutStart), toMinutes(shift.checkOutEnd))) {
    return "check_out";
  }
  return null;
}

export function reduceKiosk(state: KioskState, event: KioskEvent): KioskState {
  switch (event.type) {
    case "camera_ready":
      return { ...state, camera: "ready" };
    case "camera_error":
      return { ...state, camera: "error" };
    case "ws_connecting":
      return { ...state, socket: "connecting" };
    case "ws_open":
      return { ...state, socket: "open" };
    case "ws_close":
      return { ...state, socket: "closed" };
    case "greeting_done":
      return { ...state, greeting: null, hint: null, faceBox: null };
    case "message": {
      // While a greeting is showing we've paused capture; ignore any in-flight
      // result so a stale "unknown" can't flicker over the welcome message.
      if (state.greeting) return state;
      const msg = event.message;
      const bbox = "bbox" in msg ? msg.bbox : null;
      switch (msg.status) {
        case "recognized":
          return {
            ...state,
            greeting: {
              name: msg.name,
              message: attendanceMessage(msg.attendance),
              kind: attendanceKind(msg.attendance),
            },
            hint: null,
            faceBox: bbox,
          };
        case "unknown":
          return { ...state, hint: "Không tìm thấy khuôn mặt", faceBox: bbox };
        case "spoof":
          return { ...state, hint: null, faceBox: null };
        case "no_face":
          return { ...state, hint: null, faceBox: null };
        case "error":
          return {
            ...state,
            hint: "Hệ thống đang bận, thử lại sau giây lát",
            faceBox: null,
          };
        default:
          return state;
      }
    }
    default:
      return state;
  }
}

/** The single phase the UI renders, resolved from the two resources + greeting. */
export function kioskPhase(state: KioskState): KioskPhase {
  if (state.camera === "error") return "camera_error";
  if (state.greeting) return "recognized";
  if (state.camera !== "ready") return "initializing";
  if (state.socket !== "open") return "disconnected";
  return "scanning";
}

/** Frames are only worth sending when the camera is live and the socket is open. */
export function shouldCapture(state: KioskState): boolean {
  return (
    state.camera === "ready" && state.socket === "open" && state.greeting === null
  );
}

/** http(s) API base → ws(s) recognition endpoint for this client. */
export function recognitionWsUrl(apiBaseUrl: string, clientId: string): string {
  const wsBase = apiBaseUrl.replace(/^http/, "ws").replace(/\/+$/, "");
  return `${wsBase}/ws/recognition/${encodeURIComponent(clientId)}`;
}

/** Build the kiosk registration deep-link. Query params carry the employee
 *  name + code to the /kiosk page; URLSearchParams safely preserves spaces,
 *  unicode, and separators in either value. */
export function enrollmentUrl(name: string, empCode: string): string {
  const params = new URLSearchParams({
    mode: "register",
    name,
    emp_code: empCode,
  });
  return `/kiosk?${params.toString()}`;
}
