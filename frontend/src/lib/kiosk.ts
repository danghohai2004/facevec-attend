// Pure kiosk logic: WebSocket message contract, view state machine, and the
// Vietnamese copy shown to the person standing at the camera. No DOM here so it
// stays trivially testable — see kiosk.test.ts for the reducer self-check.

/** Messages the backend pushes over /ws/recognition/{client_id} (see pipeline.py). */
export type RecognitionMessage =
  | {
      status: "recognized";
      emp_id: number;
      name: string;
      attendance: string;
      timestamp: string;
    }
  | { status: "unknown"; timestamp: string }
  | { status: "no_face"; timestamp: string }
  | { status: "spoof"; timestamp: string }
  | { status: "error"; detail: string };

/**
 * Two async resources drive the screen independently: the camera and the
 * socket. Track each rather than flattening early — the derived `kioskPhase`
 * picks the one that matters for what the user sees.
 */
export interface KioskState {
  camera: "pending" | "ready" | "error";
  socket: "connecting" | "open" | "closed";
  greeting: { name: string; message: string } | null;
  hint: string | null;
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
      return { ...state, greeting: null, hint: null };
    case "message": {
      // While a greeting is showing we've paused capture; ignore any in-flight
      // result so a stale "unknown" can't flicker over the welcome message.
      if (state.greeting) return state;
      const msg = event.message;
      switch (msg.status) {
        case "recognized":
          return {
            ...state,
            greeting: {
              name: msg.name,
              message: attendanceMessage(msg.attendance),
            },
            hint: null,
          };
        case "unknown":
          return { ...state, hint: "Không nhận diện được khuôn mặt" };
        case "spoof":
          return { ...state, hint: "Vui lòng nhìn thẳng vào camera" };
        case "no_face":
          return { ...state, hint: null };
        case "error":
          return { ...state, hint: "Hệ thống đang bận, thử lại sau giây lát" };
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
