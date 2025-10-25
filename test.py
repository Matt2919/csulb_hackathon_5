import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
mp_face_detection = mp.solutions.face_detection

##stuff for cry cry and nya nya
prev = -1
prevr = -1
prevl = -1
prevr2 = -1
prevl2 = -1
lm1 = lm2 = lm3 = lm4 = None
rm1 = rm2 = rm3 = rm4 = None


cap = cv2.VideoCapture(0)
with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    results2 = face_detection.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ##detect for cry cry emote
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 2:
        # Separate left and right hands
        right_hand = None
        left_hand = None
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            if handedness.classification[0].label == "Right":
                right_hand = hand_landmarks
            elif handedness.classification[0].label == "Left":
                left_hand = hand_landmarks

        if right_hand and left_hand:
            # --- Cry cry check ---
            r = [right_hand.landmark[i] for i in [17, 18, 19, 20]]
            l = [left_hand.landmark[i] for i in [17, 18, 19, 20]]
            if all(abs(r[i].z - r[i + 1].z) < 0.008 for i in range(3)) and \
                    all(abs(l[i].z - l[i + 1].z) < 0.008 for i in range(3)):
                if prevr == -1:
                    prevr, prevl = r[1].x, l[1].x
                else:
                    if abs(prevr - r[1].x) > 0.008 and abs(prevl - l[1].x) > 0.008:
                        print("Cry cry")
                        prevr, prevl = r[1].x, l[1].x

            # --- Nya nya check ---
            r = [right_hand.landmark[i] for i in [5, 9, 13, 17]]
            l = [left_hand.landmark[i] for i in [5, 9, 13, 17]]
            if all(abs(r[i].z - r[i + 1].z) < 0.008 for i in range(3)) and \
                    all(abs(l[i].z - l[i + 1].z) < 0.008 for i in range(3)):
                if prevr2 == -1:
                    prevr2, prevl2 = r[1].x, l[1].x
                else:
                    if abs(prevr2 - r[1].x) > 0.008 and abs(prevl2 - l[1].x) > 0.008:
                        print("Nya nya")
                        prevr2, prevl2 = r[1].x, l[1].x

                #print(f"Landmark: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    if results2.detections:
        for detection in results2.detections:
            mp_drawing.draw_detection(image, detection)
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()