import os
import time
import math
import urllib.request
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
MODEL_NAME = "face_landmarker.task"

alpha = 0.15
score = 0.0
history = deque(maxlen=50)
best_sides = {'LEFT': 0, 'RIGHT': 0}
mask_active = False

stats = {
    'symmetry': 0, 'structure': 0, 'eyes': 0, 
    'jaw_cheeks': 0, 'lips': 0, 'brows': 0, 'skin': 0,
    'cheeks_depth': 0, 'forehead': 0, 
    'gonial_angle': 0, 'e_line': 0, 'nose_proj': 0,
    'mewing': 0, 'eye_spacing': 0, 'canthal_tilt_deg': 0
}

archetype = "Scanning..."
tilt_type = "Analyzing..."
mode = "FRONTAL"
latest_res = None

def download_model():
    if not os.path.exists(MODEL_NAME):
        print(f"Downloading {MODEL_NAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_NAME)

def callback(result, img, ts):
    global latest_res
    latest_res = result

def to_px(lms, w, h):
    return np.array([(lm.x * w, lm.y * h) for lm in lms])

def to_3d(lms, w, h):
    return np.array([(lm.x * w, lm.y * h, lm.z * w) for lm in lms])

def get_angle(p1, p2, p3):
    v1 = p1 - p2
    v2 = p3 - p2
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

def dist_line(p, l1, l2):
    v1 = np.append(l2 - l1, 0)
    v2 = np.append(l1 - p, 0)
    return np.linalg.norm(np.cross(v1, v2)) / np.linalg.norm(v1)

def get_pose(lms):
    nose = lms[1].x
    l_ear = lms[234].x
    r_ear = lms[454].x
    width = r_ear - l_ear
    if width == 0: return "FRONTAL"
    
    ratio = (nose - l_ear) / width
    if ratio < 0.25: return "PROFILE_RIGHT" 
    if ratio > 0.75: return "PROFILE_LEFT"
    return "FRONTAL"

def check_skin(img, lms, w, h):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    centers = [151, 118, 347] 
    var_sum = 0
    count = 0
    
    for idx in centers:
        lm = lms[idx]
        cx, cy = int(lm.x * w), int(lm.y * h)
        d = 24 
        x1, y1 = max(0, cx - d), max(0, cy - d)
        x2, y2 = min(w, cx + d), min(h, cy + d)
        roi = gray[y1:y2, x1:x2]
        if roi.size > 0:
            var_sum += cv2.Laplacian(roi, cv2.CV_64F).var()
            count += 1
            
    avg = var_sum / count if count > 0 else 500
    return max(0, min(100, 100 - (avg - 50) * 0.25))

def get_archetype(data, cj_ratio, f_ratio, m):
    if m != "FRONTAL": return "Profile View"
    if cj_ratio < 1.15 and data['jaw_cheeks'] > 70: return "Hunter"
    if f_ratio > 1.5: return "Royal"
    if data['eyes'] > 85 and data['jaw_cheeks'] > 80: return "Model"
    if data['skin'] > 85 and data['symmetry'] > 90: return "Idol"
    if cj_ratio > 1.35: return "Heart"
    return "Balanced"

def update_best_side(m, s):
    global best_sides
    if m == "PROFILE_LEFT" and s > best_sides['LEFT']:
        best_sides['LEFT'] = s
    elif m == "PROFILE_RIGHT" and s > best_sides['RIGHT']:
        best_sides['RIGHT'] = s
            
    if best_sides['LEFT'] == 0 and best_sides['RIGHT'] == 0: return "Unknown"
    if best_sides['LEFT'] > best_sides['RIGHT']: return "LEFT"
    if best_sides['RIGHT'] > best_sides['LEFT']: return "RIGHT"
    return "Equal"

def process_face(lms, blend, img, w, h):
    pts = to_px(lms, w, h)
    pts3 = to_3d(lms, w, h)
    m = get_pose(lms)
    data = stats.copy()
    tilt = "Neutral"
    
    if m == "FRONTAL":
        center = pts[1][0] 
        l_w = np.ptp(pts[pts[:, 0] < center][:, 0]) if np.any(pts[:, 0] < center) else 1
        r_w = np.ptp(pts[pts[:, 0] > center][:, 0]) if np.any(pts[:, 0] > center) else 1
        ratio = min(l_w, r_w) / max(l_w, r_w)
        data['symmetry'] = max(0, 100 * (1.0 - (1.0-ratio)*0.8))

        d1 = np.linalg.norm(pts[10] - pts[9]) * 1.5 
        d2 = np.linalg.norm(pts[9] - pts[2])
        d3 = np.linalg.norm(pts[2] - pts[152])
        avg = (d1 + d2 + d3) / 3
        err = (abs(d1 - avg) + abs(d2 - avg) + abs(d3 - avg)) / avg
        data['structure'] = max(0, 100 - (err * 60))

        cw = np.linalg.norm(pts[454] - pts[234])
        jw = np.linalg.norm(pts[172] - pts[397])
        cj = cw / jw if jw > 0 else 0
        
        diff_l = pts3[205][2] - pts3[50][2]
        diff_r = pts3[425][2] - pts3[280][2]
        hollow = 50 + ((diff_l + diff_r) / 2 * 1000)
        data['cheeks_depth'] = max(0, min(100, hollow))
        data['jaw_cheeks'] = (max(0, 100 - abs(cj - 1.25) * 80) * 0.6 + data['cheeks_depth'] * 0.4)
        
        fh_h = np.linalg.norm(pts[10] - pts[9])
        face_h = np.linalg.norm(pts[10] - pts[152])
        v_rat = fh_h / face_h if face_h > 0 else 0
        
        fh_w = np.linalg.norm(pts[103] - pts[332])
        ch_w = np.linalg.norm(pts[234] - pts[454])
        w_rat = fh_w / ch_w if ch_w > 0 else 0
        data['forehead'] = (max(0, 100 - abs(v_rat - 0.33) * 400) + max(0, 100 - abs(w_rat - 0.95) * 200)) / 2

        data['skin'] = check_skin(img, lms, w, h)

        dy_l = pts[133][1] - pts[33][1]
        dx_l = pts[33][0] - pts[133][0]
        deg_l = math.degrees(math.atan2(dy_l, dx_l))
        
        dy_r = pts[362][1] - pts[263][1]
        dx_r = pts[263][0] - pts[362][0]
        deg_r = math.degrees(math.atan2(dy_r, dx_r))
        
        avg_tilt = (deg_l + deg_r) / 2
        data['canthal_tilt_deg'] = avg_tilt
        
        if avg_tilt > 4: tilt = "Hunter Eyes"
        elif avg_tilt >= 0: tilt = "Neutral"
        else: tilt = "Negative Tilt"

        t_bonus = 100 if avg_tilt > 2 else (80 if avg_tilt > 0 else 60)
        
        w_eye = (np.linalg.norm(pts[33] - pts[133]) + np.linalg.norm(pts[362] - pts[263])) / 2
        inter = np.linalg.norm(pts[133] - pts[362])
        esr = inter / w_eye if w_eye > 0 else 1.0
        data['eye_spacing'] = max(0, 100 - abs(esr - 1.0) * 80)
        
        data['eyes'] = (t_bonus * 0.4 + data['eye_spacing'] * 0.6)
        data['lips'] = 85 

        final = (data['symmetry'] * 0.15 + data['structure'] * 0.15 + data['cheeks_depth'] * 0.10 +
                 data['jaw_cheeks'] * 0.15 + data['forehead'] * 0.10 + data['eyes'] * 0.20 + data['skin'] * 0.15)
        
        return min(99, max(1, final)), data, m, cj, 0, tilt

    else:
        if m == "PROFILE_RIGHT":
            ear, jaw, chin, nose, lip = 454, 397, 152, 1, 17
        else:
            ear, jaw, chin, nose, lip = 234, 172, 152, 1, 17

        ang = get_angle(pts[ear], pts[jaw], pts[chin])
        data['gonial_angle'] = max(0, 100 - abs(ang - 125) * 2)
        data['e_line'] = max(0, 100 - (dist_line(pts[lip], pts[nose], pts[chin]) * 0.5))
        data['nose_proj'] = 85

        j_dx = pts[chin][0] - pts[jaw][0]
        j_dy = pts[chin][1] - pts[jaw][1]
        slope = math.degrees(math.atan2(j_dy, abs(j_dx)))
        
        if 10 <= slope <= 25:
            data['mewing'] = 100 - abs(slope - 18) * 1.5
        elif slope < 10:
            data['mewing'] = 90
        else:
            data['mewing'] = max(0, 100 - (slope - 25) * 3)

        final = (data['gonial_angle'] * 0.25 + data['e_line'] * 0.25 + data['nose_proj'] * 0.15 + data['mewing'] * 0.35)
        return min(99, max(1, final)), data, m, 0, ang, "N/A"

def draw_watermark(img, s):
    if s >= 90:
        h, w, _ = img.shape
        txt = "A D A M"
        font = cv2.FONT_HERSHEY_TRIPLEX
        sz = cv2.getTextSize(txt, font, 2.0, 3)[0]
        x = (w - sz[0]) // 2
        y = h - 60
        for i in range(1, 4):
            cv2.putText(img, txt, (x, y), font, 2.0, (0, 100, 255), 3 + i*2, cv2.LINE_AA)
        cv2.putText(img, txt, (x, y), font, 2.0, (0, 215, 255), 3, cv2.LINE_AA)

def draw_graph(img, hist):
    if len(hist) < 2: return
    h, w, _ = img.shape
    gw, gh = 200, 100
    x0, y0 = w - gw - 20, h - gh - 20
    
    over = img.copy()
    cv2.rectangle(over, (x0, y0), (x0+gw, y0+gh), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.5, img, 0.5, 0, img)
    
    pts = []
    for i, val in enumerate(hist):
        px = int(x0 + (i / len(hist)) * gw)
        norm = max(0, min(1, (val - 50) / 50))
        py = int((y0 + gh) - (norm * gh))
        pts.append((px, py))
    
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts)], False, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, "LIVE SCORE", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

def draw_mask(img, lms, w, h):
    pts = to_px(lms, w, h).astype(int)
    c = (0, 215, 255) 
    pairs = [(33, 133), (362, 263), (10, 1), (1, 61), (1, 291), (234, 152), (454, 152), (61, 291), (70, 300)]
    for p1, p2 in pairs:
        cv2.line(img, tuple(pts[p1]), tuple(pts[p2]), c, 1)

def draw_hud(img, data, m, arch, tilt, side):
    over = img.copy()
    cv2.rectangle(over, (0, 0), (320, 650), (0, 0, 0), -1)
    cv2.addWeighted(over, 0.75, img, 0.25, 0, img)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 35
    
    cv2.putText(img, f"MODE: {m}", (10, y), font, 0.5, (0, 255, 255), 1)
    y += 25
    cv2.putText(img, f"TYPE: {arch}", (10, y), font, 0.5, (200, 200, 200), 1)
    y += 25
    cv2.putText(img, f"BEST SIDE: {side}", (10, y), font, 0.5, (0, 255, 0), 1)
    y += 35
    
    if m == "FRONTAL":
        items = [
            ("Symmetry", data['symmetry'], "Face Balance"),
            ("Structure", data['structure'], "Golden Ratio"),
            ("Eyes", data['eyes'], "Tilt & Spacing"),
            ("Cheeks", data['cheeks_depth'], "Hollow Depth"),
            ("Jawline", data['jaw_cheeks'], "Shape & Width"),
            ("Skin", data['skin'], "Texture Quality"),
            ("Forehead", data['forehead'], "Vertical Ratio")
        ]
    else:
        items = [
            ("Mewing", data['mewing'], "Tongue Posture"),
            ("Gonial", data['gonial_angle'], "Jaw Angle"),
            ("Harmony", data['e_line'], "Lips to E-Line"),
            ("Nose", data['nose_proj'], "Projection"),
        ]
        
    for lbl, val, desc in items:
        cv2.putText(img, lbl, (10, y), font, 0.5, (255, 255, 255), 1)
        cv2.putText(img, desc, (10, y+15), font, 0.35, (150, 150, 150), 1)
        
        bx = 120
        cv2.rectangle(img, (bx, y-8), (bx+160, y-2), (50, 50, 50), -1)
        
        col = (0, 0, 255)
        if val > 85: col = (0, 255, 0)
        elif val > 50: col = (0, 255, 255)
        
        fill = int((val / 100) * 160)
        cv2.rectangle(img, (bx, y-8), (bx + fill, y-2), col, -1)
        cv2.putText(img, f"{int(val)}", (bx + 165, y), font, 0.4, (255, 255, 255), 1)
        y += 45

def draw_rect(img, x, y, w, h, s):
    l = int(min(w, h) * 0.25)
    c = (0, 255, 0) if s >= 85 else ((0, 215, 255) if s >= 60 else (0, 0, 255))
    if s >= 90: c = (0, 215, 255)
    
    pts = [
        ((x, y), (x + l, y)), ((x, y), (x, y + l)),
        ((x + w, y), (x + w - l, y)), ((x + w, y), (x + w, y + l)),
        ((x, y + h), (x + l, y + h)), ((x, y + h), (x, y + h - l)),
        ((x + w, y + h), (x + w - l, y + h)), ((x + w, y + h), (x + w, y + h - l))
    ]
    for p1, p2 in pts:
        cv2.line(img, p1, p2, c, 2)
    
    txt = f"{int(s)}"
    font = cv2.FONT_HERSHEY_DUPLEX
    tw = cv2.getTextSize(txt, font, 1.3, 2)[0][0]
    cx = x + w // 2
    by = y + h + 20
    
    cv2.rectangle(img, (cx - 45, by), (cx + 45, by + 55), (0, 0, 0), -1)
    cv2.rectangle(img, (cx - 45, by), (cx + 45, by + 55), c, 1)
    cv2.putText(img, txt, (cx - tw // 2, by + 40), font, 1.3, c, 2, cv2.LINE_AA)

def main():
    download_model()
    global score, stats, archetype, mode, tilt_type, mask_active
    
    print("Starting camera... Press 'q' to quit, 'm' for mask.")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_NAME),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=callback,
        num_faces=1)

    with vision.FaceLandmarker.create_from_options(opts) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret: break

            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_img, int(time.time() * 1000))

            if latest_res and latest_res.face_landmarks:
                lms = latest_res.face_landmarks[0]
                blend = latest_res.face_blendshapes
                
                raw, det, m, cj, ang, tilt = process_face(lms, blend, frame, w, h)
                
                score = (score * (1 - alpha)) + (raw * alpha)
                history.append(score)
                
                for k, v in det.items():
                    stats[k] = (stats[k] * (1 - alpha)) + (v * alpha)
                
                mode = m
                archetype = get_archetype(stats, cj, 0, m)
                tilt_type = tilt
                best_side = update_best_side(m, score)

                if mask_active:
                    draw_mask(frame, lms, w, h)

                pts = to_px(lms, w, h).astype(int)
                if m == "FRONTAL":
                    cv2.line(frame, tuple(pts[33]), tuple(pts[133]), (100, 255, 100), 1)
                    cv2.line(frame, tuple(pts[362]), tuple(pts[263]), (100, 255, 100), 1)
                    col = (0, 255, 0) if "Hunter" in tilt else ((0, 255, 255) if "Neutral" in tilt else (0, 0, 255))
                    cv2.putText(frame, tilt, (w//2 - 50, pts[10][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                    if stats['eye_spacing'] > 0:
                        cv2.line(frame, tuple(pts[133]), tuple(pts[362]), (255, 0, 255), 1)
                else:
                    ear = 454 if m == "PROFILE_RIGHT" else 234
                    jaw = 397 if m == "PROFILE_RIGHT" else 172
                    chin = 152
                    cv2.line(frame, tuple(pts[ear]), tuple(pts[jaw]), (200, 200, 200), 2)
                    cv2.line(frame, tuple(pts[jaw]), tuple(pts[chin]), (0, 255, 255), 3)
                    cv2.putText(frame, f"Angle: {int(ang)}", (pts[jaw][0], pts[jaw][1]+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                mn, mx = np.min(pts[:, 0]), np.max(pts[:, 0])
                my, M_y = np.min(pts[:, 1]), np.max(pts[:, 1])
                draw_rect(frame, int(mn-40), int(my-70), int(mx-mn+80), int(M_y-my+80), score)
                draw_hud(frame, stats, mode, archetype, tilt, best_side)
                draw_graph(frame, history)
                draw_watermark(frame, score)

            else:
                cv2.putText(frame, "Scan Face...", (50, h//2), cv2.FONT_HERSHEY_DUPLEX, 1.5, (100, 100, 100), 2)

            cv2.imshow("Face Analysis", frame)
            k = cv2.waitKey(5) & 0xFF
            if k == ord('q'): break
            if k == ord('m'): mask_active = not mask_active

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()