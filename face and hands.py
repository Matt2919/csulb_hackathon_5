import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import platform
import time
import os


# TEXT TO SPEECH SETUP (queue, no cutoffs)

if platform.system() == "Darwin":
    engine = pyttsx3.init(driverName='nsss')   # macOS voice driver
elif platform.system() == "Windows":
    engine = pyttsx3.init(driverName='sapi5')  # Windows voice driver
else:
    engine = pyttsx3.init()                    # fallback / Linux

engine.setProperty('rate', 170)
engine.setProperty('volume', 1.0)

speech_state = {
    "queue": [],
    "is_speaking": False,
    "last_enqueued_time": {},    # per-phrase cooldown
    "enqueue_cooldown": 1.5,     # sec between same phrase
    "last_any_spoken_time": 0.0  # last time we enqueued ANY phrase at all
}

def enqueue_phrase(text: str):
    """Queue a phrase to speak later, no spam."""
    phrase = text.lower().strip()
    if not phrase:
        return
    now = time.time()

    last_t = speech_state["last_enqueued_time"].get(phrase, 0.0)
    if now - last_t < speech_state["enqueue_cooldown"]:
        return

    # don't enqueue duplicates back-to-back in queue
    if len(speech_state["queue"]) == 0 or speech_state["queue"][-1] != phrase:
        speech_state["queue"].append(phrase)
        speech_state["last_enqueued_time"][phrase] = now
        speech_state["last_any_spoken_time"] = now  # update global last spoken trigger time

def speak_next_if_free():
    """Speak 1 item fully, never interrupt mid-word."""
    if speech_state["is_speaking"]:
        return
    if not speech_state["queue"]:
        return
    phrase = speech_state["queue"].pop(0)
    speech_state["is_speaking"] = True
    try:
        engine.say(phrase)
        engine.runAndWait()
        time.sleep(0.05)
    finally:
        speech_state["is_speaking"] = False



# LOAD MODEL

clf = joblib.load("asl_model.joblib")


# MEDIAPIPE

mp_hands_api = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands_api.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


# RUNTIME STATE

state = {
    # SIX SEVEN vertical ref (left hand landmark[9].y last frame)
    "prev_vert": -1.0,

    # Cry Cry horizontal refs (right/left x)
    "prev_r_x": -1.0,
    "prev_l_x": -1.0,

    # Nya Nya depth refs (right/left z)
    "prev_r_z": -1.0,
    "prev_l_z": -1.0,

    # cooldown/overlay timers for meme overlays & re-speech
    "did67": 0,
    "didcry": 0,
    "didnya": 0,

    # last meme triggered
    "last_meme": -1,  # 1=Cry Cry, 2=Nya Nya, 3=Six Seven

    # saved coords for six/seven overlay
    "overlay_left_lm9": None,    # (x,y) normalized
    "overlay_right_lm9": None,   # (x,y) normalized

    # ASL repeat gating
    "asl_last_spoken": None,     # last ASL label said
    "asl_last_time": 0.0,        # wall time when we said that label
    "asl_cooldown": 1.5,         # min gap before repeating same label

    # long timeout override for ASL (in seconds)
    "asl_force_repeat_after": 5.0
}


# LOGGING

GESTURE_FILE = "gestures.txt"
open(GESTURE_FILE, "w").close()

def log_gesture(text: str):
    print(text)
    with open(GESTURE_FILE, "a", encoding="utf-8") as f:
        f.write(text + "\n")


# FEATURE EXTRACTION

def extract_features_from_landmarks(landmarks):
    feats = []
    for lm in landmarks:
        feats.extend([lm.x, lm.y, lm.z])
    return np.array(feats, dtype=float)


# RGBA PASTE HELPER (stickers)

def paste_rgba_centered(frame, sticker_path, cx, cy, size):
    """Paste transparent PNG centered at (cx, cy). No mirror flip."""
    if not os.path.exists(sticker_path):
        return
    sticker = cv2.imread(sticker_path, cv2.IMREAD_UNCHANGED)
    if sticker is None or sticker.size == 0:
        return

    sticker = cv2.resize(sticker, (size, size), interpolation=cv2.INTER_AREA)

    ih, iw, _ = frame.shape
    h, w = sticker.shape[:2]

    x1 = max(cx - w // 2, 0)
    y1 = max(cy - h // 2, 0)
    x2 = min(x1 + w, iw)
    y2 = min(y1 + h, ih)

    sticker_roi = sticker[: (y2 - y1), : (x2 - x1)]
    if sticker_roi.size == 0:
        return

    bgr_roi = frame[y1:y2, x1:x2]

    if sticker_roi.shape[2] == 4:
        alpha = sticker_roi[:, :, 3:4] / 255.0
        color = sticker_roi[:, :, :3]
        bgr_roi[:] = (bgr_roi * (1 - alpha) + color * alpha).astype(np.uint8)
    else:
        bgr_roi[:] = sticker_roi


# CAMERA

cap = cv2.VideoCapture(0)


# MAIN LOOP

while True:
    ok, frame = cap.read()
    if not ok:
        print("camera not giving frame, skipping")
        continue

    # selfie style flip
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    face_results = face_detection.process(rgb)

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    # gather left/right hands
    left_hand_landmarks = None
    right_hand_landmarks = None
    hand_info = []
    if results.multi_hand_landmarks and results.multi_handedness:
        hand_info = list(zip(results.multi_hand_landmarks, results.multi_handedness))
        for hand_landmarks, handedness in hand_info:
            side = handedness.classification[0].label  # "Left" or "Right"
            if side == "Left" and left_hand_landmarks is None:
                left_hand_landmarks = hand_landmarks
            elif side == "Right" and right_hand_landmarks is None:
                right_hand_landmarks = hand_landmarks

    # helper: decide if we *truly* have two distinct hands for memes
    def really_two_hands():
        # must have 2+ hands AND both labels present
        if num_hands < 2:
            return False
        if left_hand_landmarks is None or right_hand_landmarks is None:
            return False

        # make sure they aren't actually the same physical blob mislabeled:
        # measure horizontal distance between hand landmark[9] x-coords.
        l9 = left_hand_landmarks.landmark[9]
        r9 = right_hand_landmarks.landmark[9]
        # if hands are too close (like same hand flickering labels),
        # don't trust it.
        return abs(l9.x - r9.x) > 0.08  # require ~8% of frame width apart

    two_real_hands = really_two_hands()


    # ASL MODE (exactly 1 hand, OR two hands are NOT "real")
    # Include "or not two_real_hands"? Because sometimes
    # Mediapipe flickers "Left/Right" for one hand, and we
    # don't want that to kill ASL speaking.

    pred_label = None
    pred_prob = 0.0

    run_asl = (num_hands == 1) or (num_hands >= 2 and not two_real_hands)

    if run_asl and results.multi_hand_landmarks:
        # classify first visible hand
        first_landmarks = results.multi_hand_landmarks[0].landmark
        feats = extract_features_from_landmarks(first_landmarks)

        probs = clf.predict_proba([feats])[0]
        classes = clf.classes_
        best_idx = int(np.argmax(probs))
        pred_label = classes[best_idx]
        pred_prob = float(probs[best_idx])

        # draw skeleton (teaching mode)
        mp_drawing.draw_landmarks(
            frame,
            results.multi_hand_landmarks[0],
            mp_hands_api.HAND_CONNECTIONS
        )

        # should we speak this letter/word?
        now = time.time()
        confident = pred_prob > 0.4

        # normal rule: don't repeat same label too fast
        normal_gate = False
        if state["asl_last_spoken"] is None:
            normal_gate = True
        elif pred_label != state["asl_last_spoken"]:
            normal_gate = True
        elif now - state["asl_last_time"] > state["asl_cooldown"]:
            normal_gate = True

        # force repeat safety valve:
        # if it's been >asl_force_repeat_after seconds since ANY phrase was last enqueued,
        # allow it no matter what so we don't "go silent".
        quiet_too_long = (now - speech_state["last_any_spoken_time"] >
                          state["asl_force_repeat_after"])

        if confident and (normal_gate or quiet_too_long):
            log_gesture(pred_label)
            enqueue_phrase(pred_label)
            state["asl_last_spoken"] = pred_label
            state["asl_last_time"] = now

    # MEME MODE (only if we definitely have two distinct hands)

    if two_real_hands and hand_info:
        # ----- Cry Cry -----
        rm1 = rm2 = rm3 = rm4 = None
        lm1 = lm2 = lm3 = lm4 = None

        for hand_landmarks, handedness in hand_info:
            side = handedness.classification[0].label
            if side == "Right":
                rm1 = hand_landmarks.landmark[17]
                rm2 = hand_landmarks.landmark[18]
                rm3 = hand_landmarks.landmark[19]
                rm4 = hand_landmarks.landmark[20]
            elif side == "Left":
                lm1 = hand_landmarks.landmark[17]
                lm2 = hand_landmarks.landmark[18]
                lm3 = hand_landmarks.landmark[19]
                lm4 = hand_landmarks.landmark[20]

        if (rm1 and rm2 and rm3 and rm4 and lm1 and lm2 and lm3 and lm4):
            pinkies_flat_right = (abs(rm1.z - rm2.z) < 0.007 and
                                  abs(rm2.z - rm3.z) < 0.007 and
                                  abs(rm3.z - rm4.z) < 0.007)
            pinkies_flat_left  = (abs(lm1.z - lm2.z) < 0.007 and
                                  abs(lm2.z - lm3.z) < 0.007 and
                                  abs(lm3.z - lm4.z) < 0.007)

            if pinkies_flat_right and pinkies_flat_left:
                if state["prev_r_x"] < 0:
                    state["prev_r_x"] = rm2.x
                    state["prev_l_x"] = lm2.x
                else:
                    moved_r = abs(state["prev_r_x"] - rm2.x) > 0.007
                    moved_l = abs(state["prev_l_x"] - lm2.x) > 0.007
                    if moved_r and moved_l:
                        if state["didcry"] <= 0:
                            log_gesture("Cry Cry")
                            enqueue_phrase("cry cry")
                            state["didcry"] = 30
                            state["last_meme"] = 1
                        else:
                            state["didcry"] = 30
                            state["last_meme"] = 1
                        state["prev_r_x"] = rm2.x
                        state["prev_l_x"] = lm2.x

        # ----- Nya Nya -----
        rm1 = rm2 = rm3 = rm4 = None
        lm1 = lm2 = lm3 = lm4 = None

        for hand_landmarks, handedness in hand_info:
            side = handedness.classification[0].label
            if side == "Right":
                rm1 = hand_landmarks.landmark[5]
                rm2 = hand_landmarks.landmark[9]
                rm3 = hand_landmarks.landmark[13]
                rm4 = hand_landmarks.landmark[17]
            elif side == "Left":
                lm1 = hand_landmarks.landmark[5]
                lm2 = hand_landmarks.landmark[9]
                lm3 = hand_landmarks.landmark[13]
                lm4 = hand_landmarks.landmark[17]

        if (rm1 and rm2 and rm3 and rm4 and lm1 and lm2 and lm3 and lm4):
            stack_right = (abs(rm1.z - rm2.z) < 0.007 and
                           abs(rm2.z - rm3.z) < 0.007 and
                           abs(rm3.z - rm4.z) < 0.007)
            stack_left  = (abs(lm1.z - lm2.z) < 0.007 and
                           abs(lm2.z - lm3.z) < 0.007 and
                           abs(lm3.z - lm4.z) < 0.007)

            if stack_right and stack_left:
                if state["prev_r_z"] < 0:
                    state["prev_r_z"] = rm2.z
                    state["prev_l_z"] = lm2.z
                else:
                    moved_rz = abs(state["prev_r_z"] - rm2.z) > 0.007
                    moved_lz = abs(state["prev_l_z"] - lm2.z) > 0.007
                    if moved_rz and moved_lz:
                        if state["didnya"] <= 0:
                            log_gesture("Nya Nya")
                            enqueue_phrase("nya nya")
                            state["didnya"] = 30
                            state["last_meme"] = 2
                        else:
                            state["didnya"] = 30
                            state["last_meme"] = 2
                        state["prev_r_z"] = rm2.z
                        state["prev_l_z"] = lm2.z

        # ----- SIX SEVEN -----
        # extra strict: only fire if still two_real_hands
        if left_hand_landmarks and right_hand_landmarks:
            left_lm9  = left_hand_landmarks.landmark[9]
            right_lm9 = right_hand_landmarks.landmark[9]

            # cache for overlay
            state["overlay_left_lm9"]  = (left_lm9.x,  left_lm9.y)
            state["overlay_right_lm9"] = (right_lm9.x, right_lm9.y)

            if state["prev_vert"] < 0:
                state["prev_vert"] = left_lm9.y
            else:
                moved_vert = abs(left_lm9.y - state["prev_vert"]) > 0.05
                apart_vert = abs(right_lm9.y - left_lm9.y) > 0.05
                if moved_vert and apart_vert:
                    if state["did67"] <= 0:
                        log_gesture("SIX SEVEN")
                        enqueue_phrase("six seven")
                        state["did67"] = 20
                        state["last_meme"] = 3
                    else:
                        state["did67"] = 20
                        state["last_meme"] = 3
                state["prev_vert"] = left_lm9.y


    # UPDATE TIMERS

    if state["didcry"] > 0:
        state["didcry"] -= 1
    if state["didnya"] > 0:
        state["didnya"] -= 1
    if state["did67"] > 0:
        state["did67"] -= 1

 
    # DRAW OVERLAYS

    ih, iw, _ = frame.shape

    # SIX / SEVEN stickers:
    # left hand -> six.png
    # right hand -> seven.png
    if (state["did67"] > 0 and
        state["last_meme"] == 3 and
        state["overlay_left_lm9"]  is not None and
        state["overlay_right_lm9"] is not None):

        lx_norm, ly_norm = state["overlay_left_lm9"]
        rx_norm, ry_norm = state["overlay_right_lm9"]

        lx = int(lx_norm * iw)
        ly = int(ly_norm * ih) - 300
        rx = int(rx_norm * iw)
        ry = int(ry_norm * ih) - 300

        paste_rgba_centered(frame, "six.png",   lx, ly, size=200)
        paste_rgba_centered(frame, "seven.png", rx, ry, size=200)

    # Cry Cry overlay (crying_emoji.png near eye)
    if state["didcry"] > 0 and state["last_meme"] == 1 and face_results.detections:
        for detection in face_results.detections:
            kp = detection.location_data.relative_keypoints[2]
            cx = int(kp.x * iw)
            cy = int(kp.y * ih)
            paste_rgba_centered(frame, "crying_emoji.png", cx, cy, size=250)

    # Nya Nya overlay (anger.png above forehead)
    if state["didnya"] > 0 and state["last_meme"] == 2 and face_results.detections:
        for detection in face_results.detections:
            kp = detection.location_data.relative_keypoints[4]
            cx = int(kp.x * iw)
            cy = int(kp.y * ih) - 180
            paste_rgba_centered(frame, "anger.png", cx, cy, size=170)

    # Show ASL label/confidence when we're in ASL mode
    if run_asl and pred_label is not None:
        cv2.putText(
            frame,
            f"{pred_label} ({pred_prob:.2f})",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 0),
            3
        )

    # Show meme label ONLY in real two-hand mode
    meme_label = ""
    if state["last_meme"] == 1 and state["didcry"] > 0:
        meme_label = "😭 CRY CRY"
    elif state["last_meme"] == 2 and state["didnya"] > 0:
        meme_label = "NYA NYA"
    elif state["last_meme"] == 3 and state["did67"] > 0:
        meme_label = "SIX SEVEN"

    if meme_label and two_real_hands:
        cv2.putText(
            frame,
            meme_label,
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    cv2.imshow("ASL Live Prediction (with Memes)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    speak_next_if_free()


cap.release()
hands.close()
face_detection.close()
cv2.destroyAllWindows()
