# app.py -- Fast YOLOv8-n Flask live streamer with start/stop and stats
import os
import time
import re
import json
import datetime
import requests
import traceback
from threading import Lock
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

import urllib.parse
SUPABASE_URL = os.getenv("SUPABASE_URL")   # e.g. https://xyzcompany.supabase.co
SUPABASE_KEY = os.getenv("SUPABASE_KEY")   # anon or service_role key

USE_GEMINI_HTTP = True   # set True if you want to call Gemini via REST (needs GOOGLE_API_KEY)

# Load API key
GEMINI_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_KEY)

# The free, recommended model
GEMINI_MODEL = "gemini-2.5-pro"

def fetch_events_from_supabase(limit=200):
    """
    Fetch recent events from Supabase events table via REST.
    Returns a list of event dicts or raises/returns None on failure.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        # Supabase not configured
        return None

    try:
        # Build REST endpoint (PostgREST) for table `events`
        # We request all columns, order by timestamp desc, limit N
        # Example: GET /rest/v1/events?select=*&order=timestamp.desc&limit=200
        base = SUPABASE_URL.rstrip("/") + "/rest/v1/events"
        params = {
            "select": "*",
            "order": "timestamp.desc",
            "limit": str(limit)
        }
        url = base + "?" + urllib.parse.urlencode(params)

        headers = {
            "apikey": SUPABASE_KEY,
            "Authorization": f"Bearer {SUPABASE_KEY}",
            "Content-Type": "application/json"
        }

        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            events = resp.json()
            # Normalize timestamps to ISO strings if needed (Postgres timestamptz usually okay)
            for ev in events:
                # Ensure keys exist (optional)
                if "event_type" not in ev and "type" in ev:
                    ev["event_type"] = ev.get("type")
            return events
        else:
            print("[Supabase] fetch failed:", resp.status_code, resp.text)
            return None

    except Exception as e:
        print("[Supabase] exception fetching events:", e)
        return None


def parse_time_string(s):
    """Try to parse a time-like string 'YYYY-MM-DD HH:MM[:SS]' or 'HH:MM' (today). Return datetime or None."""
    s = s.strip()
    try:
        # full timestamp
        return datetime.datetime.fromisoformat(s)
    except Exception:
        pass
    # try HH:MM or HH:MM:SS (assume today)
    m = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", s)
    if m:
        now = datetime.datetime.now()
        hh = int(m.group(1))
        mm = int(m.group(2))
        ss = int(m.group(3) or 0)
        return datetime.datetime(now.year, now.month, now.day, hh, mm, ss)
    return None

def find_events_by_text(query, events):
    """Simple extractor: returns events that match types or camera names or times mentioned in query."""
    q = query.lower()

    # detect event types
    matches = []
    event_type_keywords = {
        "running": ["run", "running", "sprint", "sudden run"],
        "loitering": ["loiter", "loitering", "linger"],
        "pacing": ["pace", "pacing", "walking back and forth", "suspicious pacing"],
    }

    wanted_types = set()
    for et, kws in event_type_keywords.items():
        for kw in kws:
            if kw in q:
                wanted_types.add(et)

    # detect camera names by simple heuristics (Gate, Corridor, Entrance)
    wanted_cameras = []
    for ev in events:
        cam = ev.get("camera", "")
        if cam and cam.lower() in q:
            wanted_cameras.append(cam)

    # detect simple time ranges like "between HH:MM and HH:MM" or "last N minutes"
    time_from = None
    time_to = None
    m = re.search(r"between\s+(\d{1,2}:\d{2})\s+and\s+(\d{1,2}:\d{2})", q)
    if m:
        t1 = parse_time_string(m.group(1))
        t2 = parse_time_string(m.group(2))
        if t1 and t2:
            time_from = t1
            time_to = t2

    m2 = re.search(r"last\s+(\d+)\s+minute", q)
    if m2:
        minutes = int(m2.group(1))
        time_to = datetime.datetime.now()
        time_from = time_to - datetime.timedelta(minutes=minutes)

    # if user asked "today" / "yesterday"
    if "today" in q and not time_from:
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        time_from = today
        time_to = datetime.datetime.now()
    if "yesterday" in q and not time_from:
        today = datetime.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        yesterday = today - datetime.timedelta(days=1)
        time_from = yesterday
        time_to = today - datetime.timedelta(microseconds=1)

    # scan events
    for ev in events:
        ev_ts = None
        try:
            ev_ts = datetime.datetime.fromisoformat(ev["timestamp"])
        except Exception:
            pass

        # filter by type if requested
        if wanted_types and ev["event_type"] not in wanted_types:
            continue

        # filter by camera if user mentioned a camera
        if wanted_cameras and ev.get("camera") not in wanted_cameras:
            continue

        # filter by time range
        if time_from and ev_ts:
            if time_to is None:
                time_to = datetime.datetime.now()
            if not (time_from <= ev_ts <= time_to):
                continue

        matches.append(ev)

    return matches

def local_answer_from_events(query, events):
    """Produce a concise, factual reply from events using simple rules."""
    q = query.lower().strip()

    # quick common queries
    if "any" in q and ("running" in q or "run" in q):
        matches = find_events_by_text(query, events)
        if not matches:
            return "No running events found in the available data."
        # return brief summary
        lines = []
        for ev in matches:
            lines.append(f"{ev['timestamp']}: running (track {ev.get('track_id')}) at {ev.get('camera')}")
        return "Found running events:\n" + "\n".join(lines)

    if "summary" in q or "what happened" in q or "show me" in q or "events" in q:
        matches = find_events_by_text(query, events)
        if not matches:
            return "No matching events found in the available data."
        # group by type
        bytype = {}
        for ev in matches:
            bytype.setdefault(ev["event_type"], []).append(ev)
        parts = []
        for et, arr in bytype.items():
            parts.append(f"{len(arr)} {et} event(s)")
        return "Summary: " + ", ".join(parts) + ". Use a more specific query to get details."

    # fallback: list matching events for keywords
    matches = find_events_by_text(query, events)
    if matches:
        lines = [f"{ev['timestamp']}: {ev['event_type']} (track {ev.get('track_id')}) at {ev.get('camera')}" for ev in matches]
        return "\n".join(lines)

    # if nothing matched, be explicit
    return "I couldn't find any events matching your question in the available dummy data. Try asking about 'running', 'loitering', or 'pacing', or provide a time range."



# ---------------- CONFIG ----------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
ALLOWED_EXT = {"mp4", "avi", "mov", "mkv"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# SPEED/QUALITY PARAMETERS (tweak to trade off)
MODEL_WEIGHTS = "yolov8n.pt"   # small model
DETECT_WIDTH = 480             # smaller -> faster inference (480 is a good compromise)
FRAME_SKIP = 10                 # detect every Nth frame (increase for more speed)
CONF_THRESHOLD = 0.25
JPEG_QUALITY = 60              # lower -> smaller frames -> faster network transfer
OUTPUT_DISPLAY_WIDTH = 720     # width of frames sent to browser (resized for display)

# ----------------------------------------
app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "replace-with-a-secure-key"

# Global model
MODEL = None
MODEL_DEVICE = None
USE_HALF = False
MODEL_LOCK = Lock()  # ensure single model load at once

# Controls/stats per-upload file. Example structure:
# controls["163..._video.mp4"] = {
#   "running": False,
#   "lock": Lock(),
#   "last_frame": None,
#   "last_count": 0,
#   "counts_sum": 0,
#   "counts_frames": 0
# }
controls = {}
controls_lock = Lock()

def load_model(weights=MODEL_WEIGHTS):
    global MODEL, MODEL_DEVICE, USE_HALF
    with MODEL_LOCK:
        if MODEL is None:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            MODEL_DEVICE = device
            print(f"[MODEL] Loading {weights} on {device} ... (may download weights)")
            MODEL = YOLO(weights)
            USE_HALF = (device != "cpu")
            print(f"[MODEL] Loaded. half precision on GPU? {USE_HALF}")
    return MODEL

FACE_MODEL = None
def load_face_model():
    global FACE_MODEL
    if FACE_MODEL is None:
        print("[FACE] Loading YOLOv8n-Face model...")
        FACE_MODEL = YOLO("yolov8n-face-lindevs.pt")
    return FACE_MODEL


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def ensure_control_entry(filename):
    with controls_lock:
        if filename not in controls:
            controls[filename] = {
                "running": False,
                "lock": Lock(),

                "last_frame": None,
                "last_count": 0,
                "counts_sum": 0,
                "counts_frames": 0,

                "tracks": {},

                "running_enabled": False,
                "pacing_enabled": False,
                "face_cover_enabled": False,

                # NEW
                "loitering_enabled": False
            }
    return controls[filename]



def match_tracks(tracks, detections, dist_thresh=50):
    """
    Simple centroid-based tracker.
    detections = [(x1, y1, x2, y2, conf), ...]
    Returns: list of (track_id, centroid_x)
    """
    centroids = []
    for (x1, y1, x2, y2, conf) in detections:
        cx = (x1 + x2) / 2
        centroids.append(cx)

    assigned = []
    used_ids = set()

    # match each detected centroid to nearest existing track
    for cx in centroids:
        best_id = None
        best_dist = float("inf")

        # find nearest existing track (only tracks with history)
        for tid, tdata in list(tracks.items()):
            if tid in used_ids:
                continue
            if not tdata.get("history"):
                continue
            last_cx = tdata["history"][-1][1]
            dist = abs(cx - last_cx)
            if dist < best_dist and dist < dist_thresh:
                best_id = tid
                best_dist = dist

        # if no match, create a new incremental integer track id
        if best_id is None:
            # pick smallest positive integer not in tracks
            new_id = 1
            while new_id in tracks:
                new_id += 1
            # initialize track structure
            tracks[new_id] = {
                "history": [],
                "direction_flips": 0,
                "last_direction": None,
                "is_pacing": False,
                "is_loitering": False
            }
            best_id = new_id

        used_ids.add(best_id)
        assigned.append((best_id, cx))

    return assigned



def check_pacing(track_data, flip_threshold=3, min_move=20, window_seconds=10):
    """
    Updates track_data["is_pacing"] based on movement history.
    """
    history = track_data["history"]

    # remove old entries
    cutoff = time.time() - window_seconds
    history[:] = [h for h in history if h[0] >= cutoff]

    if len(history) < 4:
        track_data["is_pacing"] = False
        return

    # compute directions
    directions = []
    for i in range(1, len(history)):
        dx = history[i][1] - history[i-1][1]
        if abs(dx) < min_move:
            continue
        directions.append("R" if dx > 0 else "L")

    if len(directions) < 3:
        track_data["is_pacing"] = False
        return

    # count flips
    flips = 0
    for i in range(1, len(directions)):
        if directions[i] != directions[i-1]:
            flips += 1

    track_data["is_pacing"] = flips >= flip_threshold


def check_loitering(track_data, window_seconds=15, movement_threshold=40):
    """
    Marks is_loitering=True if the person stays within a small movement range
    for at least `window_seconds`.
    """
    history = track_data["history"]

    # trim old entries
    cutoff = time.time() - window_seconds
    history[:] = [h for h in history if h[0] >= cutoff]

    if len(history) < 6:  # ensure enough samples
        track_data["is_loitering"] = False
        return

    xs = [h[1] for h in history]
    ys = [h[2] for h in history]

    # compute movement range
    dx = max(xs) - min(xs)
    dy = max(ys) - min(ys)

    max_movement = max(dx, dy)

    # decide
    track_data["is_loitering"] = max_movement < movement_threshold


def check_running(track_data, speed_threshold=200.0, window_seconds=0.6):
    """
    Mark track_data['is_running'] True if recent instantaneous speed
    (pixels / second) exceeds speed_threshold.

    - track_data["history"] contains tuples (ts, cx, cy).
    - We look at the last two valid samples within window_seconds and compute speed.
    - speed_threshold default is conservative; tune to your resolution/fps.
    """
    history = track_data.get("history", [])
    if len(history) < 2:
        track_data["is_running"] = False
        return

    # consider only recent samples
    cutoff = time.time() - max(window_seconds, 1.0)  # make sure we have a short lookback
    recent = [h for h in history if h[0] >= cutoff]

    if len(recent) < 2:
        track_data["is_running"] = False
        return

    # find two most recent samples with different timestamps
    a = recent[-2]
    b = recent[-1]
    dt = b[0] - a[0]
    if dt <= 0:
        track_data["is_running"] = False
        return

    dx = b[1] - a[1]
    dy = b[2] - a[2]
    dist = (dx * dx + dy * dy) ** 0.5

    speed = dist / dt  # pixels per second

    # set running flag
    track_data["is_running"] = speed >= speed_threshold
    # also store last_speed for debugging/visualization if helpful
    track_data["last_speed"] = speed



def mjpeg_generator(video_path, filename):
    model = load_model()
    ctrl = ensure_control_entry(filename)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[STREAM] cannot open:", video_path)
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    print(f"[STREAM] open {video_path} size=({orig_w}x{orig_h}) fps={fps}")

    # detection scaling
    if orig_w > DETECT_WIDTH:
        scale = DETECT_WIDTH / orig_w
        detect_w = int(orig_w * scale)
        detect_h = int(orig_h * scale)
    else:
        scale = 1.0
        detect_w = orig_w
        detect_h = orig_h

    frame_idx = 0
    last_boxes = []
    last_count = 0

    try:
        while True:
            with ctrl["lock"]:
                running = ctrl["running"]

            if running:
                ret, frame = cap.read()
                if not ret:
                    print("[STREAM] end of video.")
                    break

                frame_idx += 1
                do_detect = (frame_idx % FRAME_SKIP == 0)

                if do_detect:
                    # detection frame
                    small = cv2.resize(frame, (detect_w, detect_h)) if scale != 1.0 else frame

                    imgsz = int((detect_w + 31) // 32) * 32
                    results = model.predict(
                        source=small,
                        imgsz=imgsz,
                        conf=CONF_THRESHOLD,
                        device=MODEL_DEVICE,
                        half=USE_HALF,
                        verbose=False,
                    )

                    r = results[0]
                    boxes_np = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.array([])
                    last_boxes = []

                    for b in boxes_np:
                        x1, y1, x2, y2 = b[:4]
                        conf_score = float(b[4]) if b.shape[0] > 4 else 0.0

                        if scale != 1.0:
                            x1, y1, x2, y2 = int(x1 / scale), int(y1 / scale), int(x2 / scale), int(y2 / scale)
                        else:
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                        last_boxes.append((x1, y1, x2, y2, conf_score))

                    last_count = len(last_boxes)

                    # ----------------------------------------
                    # TRACKING + PACING + LOITERING
                    # ----------------------------------------
                    assigned = match_tracks(ctrl["tracks"], last_boxes)
                    now = time.time()

                    updated_track_ids = set()

                    for idx, (tid, cx) in enumerate(assigned):
                        t = ctrl["tracks"][tid]
                        (x1, y1, x2, y2, _) = last_boxes[idx]
                        cy = (y1 + y2) / 2

                        t["history"].append((now, cx, cy))
                        t["last_box"] = (x1, y1, x2, y2)

                        # ---- PACING, LOITERING, RUNNING (TOGGLES) ----
                        # Suspicious pacing only if enabled
                        if ctrl.get("pacing_enabled", False):
                            check_pacing(t)
                        else:
                            t["is_pacing"] = False  # ensure clean state when off

                        # --- LOITERING (toggle controlled) ---
                        if ctrl.get("loitering_enabled", False):
                            check_loitering(t)
                        else:
                            t["is_loitering"] = False


                        # Sudden running only if enabled
                        if ctrl.get("running_enabled", False):
                            check_running(t)
                        else:
                            t["is_running"] = False


                        # ----------------------------
                        # STEP 3 — YOLOv8-Face
                        # ----------------------------
                        # ---- FACE COVER DETECTION (FULLY FIXED) ----
                        if ctrl.get("face_cover_enabled", False):
                            try:
                                face_model = load_face_model()

                                # expand head region to 70%
                                hx1, hy1 = x1, y1
                                hx2 = x2
                                hy2 = y1 + int((y2 - y1) * 0.70)

                                head_crop = frame[hy1:hy2, hx1:hx2]

                                if head_crop.size == 0:
                                    t["is_face_covered"] = False
                                else:
                                    # upscale small faces
                                    head_up = cv2.resize(head_crop, None, fx=2.0, fy=2.0)

                                    # very low threshold for partially-visible faces
                                    face_res = face_model.predict(head_up, conf=0.05, verbose=False)

                                    if len(face_res) > 0 and len(face_res[0].boxes) > 0:
                                        t["is_face_covered"] = False      # face visible
                                    else:
                                        t["is_face_covered"] = True       # face hidden / covered

                            except Exception as e:
                                print("[FACE] error:", e)
                                t["is_face_covered"] = False

                        updated_track_ids.add(tid)

                # draw bounding boxes
                disp = frame.copy()
                for (x1, y1, x2, y2, conf_score) in last_boxes:
                    cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(disp, f"person {conf_score:.2f}",
                                (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1)

                # ----------------------------------------
                # STEP 4 — OVERLAY PRIORITY
                # loiter(red) > pacing(yellow) > face-covered(purple)
                # ----------------------------------------
                with ctrl["lock"]:
                    track_snapshot = {tid: dict(tdata) for tid, tdata in ctrl["tracks"].items()}

                for tid, tdata in track_snapshot.items():
                    last_box = tdata.get("last_box")
                    if not last_box:
                        continue

                    x1, y1, x2, y2 = last_box
                    is_loiter = tdata.get("is_loitering", False)
                    is_pace = tdata.get("is_pacing", False)
                    is_face_cov = tdata.get("is_face_covered", False)

                    # priority ordering
                    if is_loiter:
                        color = (0, 0, 255)     # red
                        label = f"LOITERING #{tid}"
                        thickness = 4

                    elif is_pace:
                        color = (0, 215, 255)   # yellow
                        label = f"PACING #{tid}"
                        thickness = 3

                    elif is_face_cov:
                        color = (180, 0, 200)   # purple
                        label = f"FACE COVERED #{tid}"
                        thickness = 3

                    else:
                        continue

                    cv2.rectangle(disp, (x1, y1), (x2, y2), color, thickness)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    rect_y2 = y1 - 6
                    rect_y1 = rect_y2 - (th + 10)
                    cv2.rectangle(disp, (x1, rect_y1), (x1 + tw + 10, rect_y2), color, -1)
                    cv2.putText(disp, label, (x1 + 4, rect_y2 - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (255, 255, 255), 1)

                # resize
                if OUTPUT_DISPLAY_WIDTH and disp.shape[1] > OUTPUT_DISPLAY_WIDTH:
                    scale_out = OUTPUT_DISPLAY_WIDTH / disp.shape[1]
                    disp = cv2.resize(disp, (OUTPUT_DISPLAY_WIDTH, int(disp.shape[0] * scale_out)))

                # encode jpeg
                success, jpeg = cv2.imencode('.jpg', disp, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                if not success:
                    continue
                frame_bytes = jpeg.tobytes()

                # update frame + stats
                with ctrl["lock"]:
                    ctrl["last_frame"] = frame_bytes
                    ctrl["last_count"] = last_count
                    ctrl["counts_sum"] += last_count
                    ctrl["counts_frames"] += 1

                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')

            else:
                # paused mode
                with ctrl["lock"]:
                    lf = ctrl.get("last_frame")

                if lf is not None:
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                           lf + b'\r\n')
                else:
                    placeholder = 255 * np.ones((240, 320, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "PAUSED", (40, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    success, jpeg = cv2.imencode('.jpg', placeholder)
                    if success:
                        pb = jpeg.tobytes()
                        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
                               pb + b'\r\n')

                time.sleep(0.1)

    except GeneratorExit:
        print("[STREAM] generator exit for", filename)
    except Exception:
        traceback.print_exc()
    finally:
        cap.release()
        print("[STREAM] closed", video_path)



# Routes & frontend integration

@app.route("/")
def index():
    uploads = sorted(os.listdir(app.config["UPLOAD_FOLDER"]), reverse=True)
    return render_template("index.html", uploads=uploads)

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("video")
    if file is None or file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("File extension not allowed.")
        return redirect(url_for("index"))

    ts = int(time.time())
    safe_name = f"{ts}_{os.path.basename(file.filename)}"
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], safe_name)
    file.save(input_path)
    print(f"[UPLOAD] saved to: {input_path}")

    # initialize control entry
    ctrl_entry = ensure_control_entry(safe_name)

    return redirect(url_for("view_file", filename=safe_name))

@app.route("/view/<path:filename>")
def view_file(filename):
    # simple page with upload stream controls
    uploads = sorted(os.listdir(app.config["UPLOAD_FOLDER"]), reverse=True)
    return render_template("view.html", filename=filename, uploads=uploads)

@app.route("/video_feed")
def video_feed():
    filename = request.args.get("file")
    if not filename:
        return "Missing file parameter", 400
    safe = os.path.basename(filename)
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], safe)
    if not os.path.exists(input_path):
        return "File not found", 404
    # ensure controls entry exists
    ensure_control_entry(safe)
    return Response(mjpeg_generator(input_path, safe),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/control", methods=["POST"])
def control():
    """
    POST JSON: {"file": "<filename>", "action": "start"|"stop"}
    """
    data = request.get_json(force=True)
    filename = data.get("file")
    action = data.get("action")
    if not filename or action not in ("start", "stop"):
        return jsonify({"ok": False, "error": "invalid params"}), 400
    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)
    with ctrl["lock"]:
        ctrl["running"] = (action == "start")
    return jsonify({"ok": True, "running": ctrl["running"]})


@app.route("/toggle_loitering", methods=["POST"])
def toggle_loitering():
    data = request.get_json(force=True)
    filename = data.get("file")
    enabled = data.get("enabled")

    if not filename:
        return jsonify({"ok": False, "error": "missing filename"}), 400

    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)

    with ctrl["lock"]:
        ctrl["loitering_enabled"] = bool(enabled)

    print(f"[LOITERING] loitering_enabled for {safe} -> {bool(enabled)}")

    return jsonify({"ok": True})



@app.route("/toggle_pacing", methods=["POST"])
def toggle_pacing():
    """
    Enables or disables suspicious pacing detection for a given file.
    Body: { "file": "<filename>", "enabled": true/false }
    Returns: { "ok": true }
    """
    data = request.get_json(force=True)
    filename = data.get("file")
    enabled = data.get("enabled")

    if not filename:
        return jsonify({"ok": False, "error": "missing filename"}), 400

    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)

    with ctrl["lock"]:
        ctrl["pacing_enabled"] = bool(enabled)

    print(f"[PACING] pacing_enabled for {safe} -> {bool(enabled)}")

    return jsonify({"ok": True})


@app.route("/toggle_running", methods=["POST"])
def toggle_running():
    data = request.get_json(force=True)
    filename = data.get("file")
    enabled = data.get("enabled")

    if filename is None or enabled is None:
        return jsonify({"ok": False, "error": "invalid params"}), 400

    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)

    with ctrl["lock"]:
        ctrl["running_enabled"] = bool(enabled)

    print(f"[RUNNING DETECTION] set to {enabled} for {filename}")

    return jsonify({"ok": True, "running_enabled": ctrl['running_enabled']})



@app.route("/stats")
def stats():
    filename = request.args.get("file")
    if not filename:
        return jsonify({"ok": False, "error": "missing file param"}), 400

    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)

    with ctrl["lock"]:
        # Basic people counts
        current = int(ctrl.get("last_count", 0))
        sumc = int(ctrl.get("counts_sum", 0))
        frames = int(ctrl.get("counts_frames", 0))

        # --- DETECTION STATES ---
        pacing_detected = any(t.get("is_pacing", False) for t in ctrl["tracks"].values())
        loitering_detected = any(t.get("is_loitering", False) for t in ctrl["tracks"].values())
        running_detected = any(t.get("is_running", False) for t in ctrl["tracks"].values())

        # --- TOGGLE STATES ---
        pacing_enabled = bool(ctrl.get("pacing_enabled", False))
        loitering_enabled = bool(ctrl.get("loitering_enabled", False))
        running_enabled = bool(ctrl.get("running_enabled", False))
        face_cover_enabled = bool(ctrl.get("face_cover_enabled", False))

        # --- PER-PERSON FACE COVER LIST ---
        face_status_list = [
            {"id": tid, "covered": bool(t.get("is_face_covered", False))}
            for tid, t in ctrl["tracks"].items()
        ]

        # --- OPTIONAL COUNTS ---
        pacing_count = sum(1 for t in ctrl["tracks"].values() if t.get("is_pacing", False))
        loiter_count = sum(1 for t in ctrl["tracks"].values() if t.get("is_loitering", False))
        running_count = sum(1 for t in ctrl["tracks"].values() if t.get("is_running", False))

    # Average people in the scene
    avg_people = (sumc / frames) if frames > 0 else 0.0

    return jsonify({
        "ok": True,

        # People counts
        "current_count": current,
        "avg_people": round(avg_people, 2),

        # --- Suspicious pacing ---
        "suspicious_pacing": pacing_detected,
        "pacing_enabled": pacing_enabled,
        "pacing_count": pacing_count,

        # --- Loitering ---
        "loitering_detected": loitering_detected,
        "loitering_enabled": loitering_enabled,
        "loitering_count": loiter_count,

        # --- Running ---
        "running_detected": running_detected,
        "running_enabled": running_enabled,
        "running_count": running_count,

        # --- Face Cover Detection ---
        "face_cover_enabled": face_cover_enabled,
        "face_covering_list": face_status_list
    })




@app.route("/download/<folder>/<filename>")
def download_file(folder, filename):
    if folder not in ("uploads", "outputs"):
        return "Invalid folder", 400
    directory = UPLOAD_FOLDER if folder == "uploads" else OUTPUT_FOLDER
    return send_from_directory(directory, filename, as_attachment=True)






# ---------------------------
# Updated /ask_ai route (Supabase-backed)
# ---------------------------
@app.route("/ask_ai", methods=["POST"])
def ask_ai():
    data = request.get_json()
    query = (data.get("query") or "").strip()

    if not query:
        return jsonify({"reply": "Please enter a question."})

    # Try to fetch real events from Supabase
    events = fetch_events_from_supabase(limit=400)
    if not events:
        # fallback to dummy events that already exist in file
        events = dummy_events

    # Build the system prompt with the real events (trim to avoid very large prompts)
    try:
        # Keep prompt short: include at most 150 events to avoid giant prompts
        prompt_events = events[:150] if isinstance(events, list) else events
        system_text = (
            "You are a CCTV surveillance assistant. Use ONLY the event data provided below to answer the user's question. "
            "If the user asks about something not present in the events, say so clearly. Be concise and factual.\n\n"
            "EVENTS (most recent first):\n" + json.dumps(prompt_events, indent=2)
        )
    except Exception:
        # In case events are not serializable for some reason
        system_text = "You are a CCTV surveillance assistant. Use only the provided events."

    # If Gemini key not configured, use the local responder
    if not GEMINI_KEY:
        return jsonify({"reply": local_answer_from_events(query, events)})

    # Call Gemini via official SDK (genai) as you already do
    try:
        # Build model (use model name already defined in file)
        model = genai.GenerativeModel(model_name=GEMINI_MODEL, system_instruction=system_text)

        # For the current SDK you used earlier: generate content
        response = model.generate_content(query)

        reply = response.text.strip() if response and getattr(response, "text", None) else None
        if reply:
            return jsonify({"reply": reply})
        else:
            # fallback to local summary if model returns empty
            return jsonify({"reply": local_answer_from_events(query, events)})
    except Exception as e:
        print("Gemini ERROR in Supabase ask_ai:", e)
        # As a robust fallback return a local rule-based answer built from the fetched events
        try:
            return jsonify({"reply": local_answer_from_events(query, events)})
        except Exception as e2:
            print("Local fallback also failed:", e2)
            return jsonify({"reply": "Error processing request."}), 500





if __name__ == "__main__":
    print("Starting server...")
    print(f"uploads -> {UPLOAD_FOLDER}")
    try:
        load_model()
    except Exception as e:
        print("[WARN] model preload error:", e)
    # threaded mode allows concurrent streams for dev (not for high scale)
    app.run(host="0.0.0.0", port=5000, threaded=True)


@app.route("/toggle_face_cover", methods=["POST"])
def toggle_face_cover():
    data = request.get_json(force=True)
    filename = data.get("file")
    enabled = data.get("enabled")

    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)

    with ctrl["lock"]:
        ctrl["face_cover_enabled"] = bool(enabled)

    return jsonify({"ok": True})

