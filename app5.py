# app.py -- Fast YOLOv8-n Flask live streamer with start/stop and stats
import os
import time
import traceback
from threading import Lock
from flask import Flask, request, redirect, url_for, render_template, send_from_directory, flash, Response, jsonify
import cv2
import numpy as np
from ultralytics import YOLO

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
                "tracks": {}          # NEW: per-ID tracks
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



def mjpeg_generator(video_path, filename):
    """
    Generator that yields multipart JPEG frames for MJPEG streaming.
    It updates controls[filename] stats (last_count, counts_sum, counts_frames).
    The displayed frames do NOT include the 'People: N' overlay.
    """
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

    # compute detection scale
    if orig_w > DETECT_WIDTH:
        scale = DETECT_WIDTH / orig_w
        detect_w = int(orig_w * scale)
        detect_h = int(orig_h * scale)
    else:
        scale = 1.0
        detect_w = orig_w
        detect_h = orig_h

    device_for_predict = MODEL_DEVICE
    use_half = USE_HALF

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
                    # resize for detection
                    if scale != 1.0:
                        small = cv2.resize(frame, (detect_w, detect_h))
                    else:
                        small = frame

                    imgsz = int((detect_w + 31) // 32) * 32
                    results = model.predict(
                        source=small,
                        imgsz=imgsz,
                        conf=CONF_THRESHOLD,
                        device=device_for_predict,
                        half=use_half,
                        verbose=False,
                    )

                    r = results[0]
                    boxes_np = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.array([])
                    last_boxes = []

                    for b in boxes_np:
                        x1, y1, x2, y2 = b[:4]
                        conf_score = float(b[4]) if b.shape[0] > 4 else 0.0

                        if scale != 1.0:
                            x1 = int(x1 / scale)
                            y1 = int(y1 / scale)
                            x2 = int(x2 / scale)
                            y2 = int(y2 / scale)
                        else:
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

                        last_boxes.append((x1, y1, x2, y2, conf_score))

                    last_count = len(last_boxes)

                    # ----------------------------------------------------
                    # ---- TRACKING + PACING ----
                    # ----------------------------------------------------
                    assigned = match_tracks(ctrl["tracks"], last_boxes)
                    now = time.time()

                    for idx, (tid, cx) in enumerate(assigned):
                        t = ctrl["tracks"][tid]
                        # get corresponding detection box to compute y-center
                        (x1, y1, x2, y2, _) = last_boxes[idx]
                        cy = (y1 + y2) / 2

                        # store (timestamp, x-center, y-center)
                        t["history"].append((now, cx, cy))

                        # run pacing
                        check_pacing(t)

                        # run loitering
                        check_loitering(t)

                # draw boxes
                disp = frame.copy()
                if last_boxes:
                    for (x1, y1, x2, y2, conf_score) in last_boxes:
                        cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"person {conf_score:.2f}"
                        cv2.putText(disp, label, (x1, max(15, y1 - 6)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                    (255, 255, 255), 1, cv2.LINE_AA)

                # resize for display
                if OUTPUT_DISPLAY_WIDTH and disp.shape[1] > OUTPUT_DISPLAY_WIDTH:
                    scale_out = OUTPUT_DISPLAY_WIDTH / disp.shape[1]
                    new_w = OUTPUT_DISPLAY_WIDTH
                    new_h = int(disp.shape[0] * scale_out)
                    disp = cv2.resize(disp, (new_w, new_h))

                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
                success, jpeg = cv2.imencode('.jpg', disp, encode_params)
                if not success:
                    continue

                frame_bytes = jpeg.tobytes()

                with ctrl["lock"]:
                    ctrl["last_frame"] = frame_bytes
                    ctrl["last_count"] = last_count
                    ctrl["counts_sum"] += last_count
                    ctrl["counts_frames"] += 1

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       frame_bytes + b'\r\n')

            else:
                # paused mode
                with ctrl["lock"]:
                    lf = ctrl.get("last_frame")

                if lf is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           lf + b'\r\n')
                else:
                    placeholder = 255 * np.ones((240, 320, 3), dtype=np.uint8)
                    cv2.putText(
                        placeholder, "PAUSED",
                        (40, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3
                    )
                    success, jpeg = cv2.imencode('.jpg', placeholder)
                    if success:
                        pb = jpeg.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' +
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

@app.route("/stats")
def stats():
    """
    Query params: ?file=<filename>
    Returns JSON: {"ok": True, "current_count": int, "avg_people": float, "suspicious_pacing": bool}
    """
    filename = request.args.get("file")
    if not filename:
        return jsonify({"ok": False, "error": "missing file param"}), 400

    safe = os.path.basename(filename)
    ctrl = ensure_control_entry(safe)

    with ctrl["lock"]:
        current = int(ctrl.get("last_count", 0))
        sumc = int(ctrl.get("counts_sum", 0))
        frames = int(ctrl.get("counts_frames", 0))

        # NEW — check if any track is pacing
        pacing_detected = any(
            t.get("is_pacing", False)
            for t in ctrl.get("tracks", {}).values()
        )
        loitering_detected = any(
            t.get("is_loitering", False)
            for t in ctrl.get("tracks", {}).values()
        )

    avg = (sumc / frames) if frames > 0 else 0.0

    return jsonify({
        "ok": True,
        "current_count": current,
        "avg_people": round(avg, 2),
        "suspicious_pacing": pacing_detected,
        "loitering_detected": loitering_detected
    })




@app.route("/download/<folder>/<filename>")
def download_file(folder, filename):
    if folder not in ("uploads", "outputs"):
        return "Invalid folder", 400
    directory = UPLOAD_FOLDER if folder == "uploads" else OUTPUT_FOLDER
    return send_from_directory(directory, filename, as_attachment=True)

if __name__ == "__main__":
    print("Starting server...")
    print(f"uploads -> {UPLOAD_FOLDER}")
    try:
        load_model()
    except Exception as e:
        print("[WARN] model preload error:", e)
    # threaded mode allows concurrent streams for dev (not for high scale)
    app.run(host="0.0.0.0", port=5000, threaded=True)
