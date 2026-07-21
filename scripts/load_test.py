#!/usr/bin/env python3
"""Load test: simulate 5-10 cameras sending frames over WebSocket.

Usage:
    python scripts/load_test.py --cameras 5       # 5 cameras
    python scripts/load_test.py --cameras 10      # 10 cameras
    python scripts/load_test.py --cameras 5 --duration 60 --fps 2

The script connects N WebSocket clients to ws://localhost:8000/ws/recognition/{id},
each sending JPEG frames at the configured FPS, collecting response latencies,
then prints a detailed report.

Requirements (stdlib + websockets):
    pip install websockets   (or: uv pip install websockets)
"""

import argparse
import asyncio
import io
import json
import logging
import os
import struct
import sys
import time
from dataclasses import dataclass, field
from statistics import mean, median, quantiles

# ---------------------------------------------------------------------------
# Synthetic frame generation — no real camera needed
# ---------------------------------------------------------------------------


def _make_synthetic_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Generate a minimal valid JPEG with random-ish pixel data.

    We use OpenCV if available (fast, realistic frame size).
    Falls back to a tiny hand-crafted JPEG otherwise.
    """
    try:
        import cv2
        import numpy as np

        # A random-noise frame is ~30-60 KB as JPEG — realistic webcam size
        img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 50])
        if ok:
            return buf.tobytes()
    except ImportError:
        pass

    # Fallback: minimal 1x1 JPEG (the server will decode it, find no face → "no_face")
    return (
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
        b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
        b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
        b"\x1f\x1e\x1d\x1a\x1c\x1c $.\x27 ,#\x1c\x1c(7),01444\x1f\x27"
        b"9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00\x01\x01\x01\x11\x00"
        b"\xff\xc4\x00\x1f\x00\x00\x01\x05\x01\x01\x01\x01\x01\x01\x00"
        b"\x00\x00\x00\x00\x00\x00\x00\x01\x02\x03\x04\x05\x06\x07\x08"
        b"\t\n\x0b\xff\xc4\x00\xb5\x10\x00\x02\x01\x03\x03\x02\x04\x03"
        b"\x05\x05\x04\x04\x00\x00\x01}\x01\x02\x03\x00\x04\x11\x05\x12"
        b"!1A\x06\x13Qa\x07\"q\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1"
        b"\xf0$3br\x82\t\n\x16\x17\x18\x19\x1a%&\x27()*456789:CDEFGHIJ"
        b"STUVWXYZ\xff\xda\x00\x08\x01\x01\x00\x00?\x00\xfb\xd2\x8a(\x03"
        b"\xff\xd9"
    )


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------


@dataclass
class CameraStats:
    """Per-camera statistics."""

    camera_id: str
    frames_sent: int = 0
    responses_received: int = 0
    errors: int = 0
    latencies: list[float] = field(default_factory=list)
    status_counts: dict[str, int] = field(default_factory=dict)
    send_errors: int = 0
    first_frame_at: float = 0.0
    last_frame_at: float = 0.0


# ---------------------------------------------------------------------------
# WebSocket camera simulator
# ---------------------------------------------------------------------------


async def camera_worker(
    camera_id: str,
    server_url: str,
    fps: float,
    duration: float,
    frame_data: bytes,
    stats: CameraStats,
) -> None:
    """Simulate a single camera: connect via WS, send frames, collect responses."""
    import websockets
    import websockets.exceptions

    interval = 1.0 / fps
    uri = f"{server_url}/ws/recognition/{camera_id}"
    pending_times: dict[int, float] = {}  # seq -> send_time
    seq = 0

    try:
        async with websockets.connect(uri, max_size=10 * 1024 * 1024) as ws:
            end_time = time.monotonic() + duration
            stats.first_frame_at = time.monotonic()

            async def sender():
                nonlocal seq
                while time.monotonic() < end_time:
                    t0 = time.monotonic()
                    try:
                        await ws.send(frame_data)
                        pending_times[seq] = time.time()
                        seq += 1
                        stats.frames_sent += 1
                    except Exception:
                        stats.send_errors += 1
                    elapsed = time.monotonic() - t0
                    await asyncio.sleep(max(0, interval - elapsed))

            async def receiver():
                try:
                    async for msg in ws:
                        recv_time = time.time()
                        stats.responses_received += 1
                        try:
                            data = json.loads(msg)
                            status = data.get("status", "unknown_status")
                            stats.status_counts[status] = (
                                stats.status_counts.get(status, 0) + 1
                            )
                        except json.JSONDecodeError:
                            stats.errors += 1

                        # Approximate latency: match response to the oldest pending frame
                        if pending_times:
                            oldest_seq = min(pending_times)
                            send_time = pending_times.pop(oldest_seq)
                            latency = recv_time - send_time
                            if latency > 0:
                                stats.latencies.append(latency)
                except websockets.exceptions.ConnectionClosedError:
                    pass

            # Run sender + receiver concurrently; when sender finishes wait
            # a bit for trailing responses, then cancel receiver.
            sender_task = asyncio.create_task(sender())
            receiver_task = asyncio.create_task(receiver())

            await sender_task
            # Give server time to flush remaining responses
            await asyncio.sleep(2.0)
            receiver_task.cancel()
            try:
                await receiver_task
            except asyncio.CancelledError:
                pass

            stats.last_frame_at = time.monotonic()

    except Exception as exc:
        logging.error("Camera %s connection failed: %s", camera_id, exc)
        stats.errors += 1


# ---------------------------------------------------------------------------
# System resource sampling (optional)
# ---------------------------------------------------------------------------


async def sample_resources(
    pid: int, interval: float, stop_event: asyncio.Event
) -> list[dict]:
    """Periodically sample CPU and memory of the target process."""
    samples = []
    try:
        import psutil

        proc = psutil.Process(pid)
    except (ImportError, Exception):
        return samples

    while not stop_event.is_set():
        try:
            cpu = proc.cpu_percent(interval=None)
            mem = proc.memory_info()
            samples.append(
                {
                    "time": time.time(),
                    "cpu_percent": cpu,
                    "rss_mb": mem.rss / (1024 * 1024),
                }
            )
        except Exception:
            break
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
            break
        except asyncio.TimeoutError:
            pass
    return samples


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _percentile(data: list[float], p: float) -> float:
    """Simple percentile; p in 0-100."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(s) else f
    d = k - f
    return s[f] + d * (s[c] - s[f])


def print_report(
    all_stats: list[CameraStats],
    duration: float,
    resource_samples: list[dict],
) -> None:
    total_sent = sum(s.frames_sent for s in all_stats)
    total_recv = sum(s.responses_received for s in all_stats)
    total_errors = sum(s.errors + s.send_errors for s in all_stats)
    all_latencies = []
    for s in all_stats:
        all_latencies.extend(s.latencies)

    merged_statuses: dict[str, int] = {}
    for s in all_stats:
        for k, v in s.status_counts.items():
            merged_statuses[k] = merged_statuses.get(k, 0) + v

    width = 70
    print("\n" + "=" * width)
    print("  LOAD TEST REPORT  ".center(width, "="))
    print("=" * width)

    print(f"\n{'Cameras:':<30} {len(all_stats)}")
    print(f"{'Test duration:':<30} {duration:.1f}s")
    print(f"{'Total frames sent:':<30} {total_sent}")
    print(f"{'Total responses received:':<30} {total_recv}")
    print(f"{'Response rate:':<30} {total_recv / total_sent * 100:.1f}%" if total_sent else "")
    print(f"{'Total errors:':<30} {total_errors}")
    print(
        f"{'Throughput (frames/sec):':<30} {total_sent / duration:.1f}"
        if duration
        else ""
    )

    # Latency
    print(f"\n{'─── Latency (seconds) ───'}")
    if all_latencies:
        print(f"  {'Min:':<26} {min(all_latencies):.4f}")
        print(f"  {'Mean:':<26} {mean(all_latencies):.4f}")
        print(f"  {'Median (p50):':<26} {_percentile(all_latencies, 50):.4f}")
        print(f"  {'p95:':<26} {_percentile(all_latencies, 95):.4f}")
        print(f"  {'p99:':<26} {_percentile(all_latencies, 99):.4f}")
        print(f"  {'Max:':<26} {max(all_latencies):.4f}")
    else:
        print("  No latency data collected.")

    # Status breakdown
    print(f"\n{'─── Response Status Breakdown ───'}")
    for status, count in sorted(merged_statuses.items(), key=lambda x: -x[1]):
        pct = count / total_recv * 100 if total_recv else 0
        print(f"  {status:<26} {count:>6}  ({pct:.1f}%)")

    # Per-camera summary
    print(f"\n{'─── Per-Camera Summary ───'}")
    print(f"  {'Camera':<12} {'Sent':>6} {'Recv':>6} {'Err':>5} {'p50(s)':>8} {'p95(s)':>8}")
    print(f"  {'─' * 12} {'─' * 6} {'─' * 6} {'─' * 5} {'─' * 8} {'─' * 8}")
    for s in all_stats:
        p50 = _percentile(s.latencies, 50) if s.latencies else 0
        p95 = _percentile(s.latencies, 95) if s.latencies else 0
        errs = s.errors + s.send_errors
        print(
            f"  {s.camera_id:<12} {s.frames_sent:>6} "
            f"{s.responses_received:>6} {errs:>5} "
            f"{p50:>8.4f} {p95:>8.4f}"
        )

    # Resource usage
    if resource_samples:
        cpus = [s["cpu_percent"] for s in resource_samples]
        mems = [s["rss_mb"] for s in resource_samples]
        print(f"\n{'─── Server Resource Usage ───'}")
        print(f"  {'CPU avg:':<26} {mean(cpus):.1f}%")
        print(f"  {'CPU peak:':<26} {max(cpus):.1f}%")
        print(f"  {'RAM avg:':<26} {mean(mems):.0f} MB")
        print(f"  {'RAM peak:':<26} {max(mems):.0f} MB")

    # Verdict
    print(f"\n{'─── Verdict ───'}")
    drop_rate = (1 - total_recv / total_sent) * 100 if total_sent else 0
    if all_latencies:
        p95 = _percentile(all_latencies, 95)
    else:
        p95 = float("inf")

    issues = []
    if drop_rate > 20:
        issues.append(f"⚠  High frame drop rate: {drop_rate:.1f}%")
    if p95 > 2.0:
        issues.append(f"⚠  p95 latency too high: {p95:.2f}s (target < 2s)")
    if total_errors > total_sent * 0.05:
        issues.append(f"⚠  Error rate: {total_errors / total_sent * 100:.1f}%")

    if issues:
        print("  ❌ POTENTIAL ISSUES DETECTED:")
        for issue in issues:
            print(f"     {issue}")
    else:
        print(f"  ✅ System handled {len(all_stats)} cameras well!")
        print(f"     Drop rate: {drop_rate:.1f}% | p95: {p95:.4f}s | Errors: {total_errors}")

    print("\n" + "=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def run_load_test(
    num_cameras: int,
    duration: float,
    fps: float,
    server_url: str,
    server_pid: int | None,
) -> None:
    print(f"\n🚀 Starting load test: {num_cameras} cameras, {fps} FPS, {duration}s")
    print(f"   Server: {server_url}")
    print(f"   Expected total frames: ~{int(num_cameras * fps * duration)}\n")

    frame_data = _make_synthetic_jpeg()
    print(f"   Frame size: {len(frame_data):,} bytes")

    all_stats = []
    tasks = []

    for i in range(num_cameras):
        cam_id = f"loadtest-cam-{i:02d}"
        stats = CameraStats(camera_id=cam_id)
        all_stats.append(stats)
        tasks.append(
            camera_worker(cam_id, server_url, fps, duration, frame_data, stats)
        )

    # Optional resource monitoring
    stop_event = asyncio.Event()
    resource_samples: list[dict] = []
    resource_task = None
    if server_pid:
        resource_task = asyncio.create_task(
            sample_resources(server_pid, 1.0, stop_event)
        )

    print(f"\n   ⏱  Running for {duration}s ...\n")
    t_start = time.monotonic()
    await asyncio.gather(*tasks)
    actual_duration = time.monotonic() - t_start

    stop_event.set()
    if resource_task:
        resource_samples = await resource_task

    print_report(all_stats, actual_duration, resource_samples)


def main():
    parser = argparse.ArgumentParser(
        description="Load test: simulate multiple cameras sending to the server"
    )
    parser.add_argument(
        "--cameras",
        type=int,
        default=5,
        help="Number of concurrent cameras (default: 5)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Test duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames per second per camera (default: 2)",
    )
    parser.add_argument(
        "--server",
        default="ws://localhost:8000",
        help="WebSocket server URL (default: ws://localhost:8000)",
    )
    parser.add_argument(
        "--server-pid",
        type=int,
        default=None,
        help="PID of the server process (for CPU/RAM monitoring via psutil)",
    )

    args = parser.parse_args()
    asyncio.run(
        run_load_test(args.cameras, args.duration, args.fps, args.server, args.server_pid)
    )


if __name__ == "__main__":
    main()
