import cv2
import mediapipe as mp
import numpy as np
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
did67 = -1
didcry = -1
didnya = -1
last = -1
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
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    rm1 = hand_landmarks.landmark[17]
                    rm2 = hand_landmarks.landmark[18]
                    rm3 = hand_landmarks.landmark[19]
                    rm4 = hand_landmarks.landmark[20]
                if handedness.classification[0].label == "Left":
                    lm1 = hand_landmarks.landmark[17]
                    lm2 = hand_landmarks.landmark[18]
                    lm3 = hand_landmarks.landmark[19]
                    lm4 = hand_landmarks.landmark[20]

            ##coding is my passion
                ##this looks to see if the joints are in similar z axis spots
                if all(v is not None for v in [rm1, rm2, rm3, rm4, lm1, lm2, lm3, lm4]):
                  if (abs(rm1.z-rm2.z) <0.006):
                        if abs(rm2.z-rm3.z) < 0.006:
                            if abs(rm3.z-rm4.z) < 0.006:
                                if (abs(lm1.z - lm2.z) < 0.006):
                                    if abs(lm2.z - lm3.z) < 0.006:
                                        if abs(lm3.z - lm4.z) < 0.006:
                                            #print("similar")
                                            if prevr == -1:
                                                prevr = rm2.x
                                                prevl = lm2.x
                                            else:
                                                #print(prevr,rm2.x)
                                                if abs(prevr - rm2.x) > 0.006:
                                                    if abs(prevl - lm2.x) > 0.006:
                                                        print("Cry cry")
                                                        last = 1
                                                        didcry = 30
                                                        prevr = rm2.x
                                                        prevl = lm2.x


    #detect for nya nya, uses the same code as before, but like dif landmarks
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness):
                if handedness.classification[0].label == "Right":
                    rm1 = hand_landmarks.landmark[5]
                    rm2 = hand_landmarks.landmark[9]
                    rm3 = hand_landmarks.landmark[13]
                    rm4 = hand_landmarks.landmark[17]
                if handedness.classification[0].label == "Left":
                    lm1 = hand_landmarks.landmark[5]
                    lm2 = hand_landmarks.landmark[9]
                    lm3 = hand_landmarks.landmark[13]
                    lm4 = hand_landmarks.landmark[17]
    #coding is my passion
                if all(v is not None for v in [rm1, rm2, rm3, rm4, lm1, lm2, lm3, lm4]):
                    if (abs(rm1.z - rm2.z) < 0.006):
                        if abs(rm2.z - rm3.z) < 0.006:
                            if abs(rm3.z - rm4.z) < 0.006:
                                if (abs(lm1.z - lm2.z) < 0.006):
                                    if abs(lm2.z - lm3.z) < 0.006:
                                        if abs(lm3.z - lm4.z) < 0.006:
                                            # print("similar")
                                            if prevr == -1:
                                                prevr2 = rm2.z
                                                prevl2 = lm2.z
                                            else:
                                                # print(prevr,rm2.x)
                                                if abs(prevr2 - rm2.z) > 0.006:
                                                    if abs(prevl2 - lm2.z) > 0.006:
                                                        didnya = 30
                                                        print("nya nya")
                                                        last = 2
                                                        prevr2 = rm2.x
                                                        prevl2 = lm2.x




    ##67 stuff lol
    didcry = didcry -1
    didnya = didnya -1
    did67 = did67 -1
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness):
            #did67 = did67 - 1
            if handedness.classification[0].label == "Right":
                rm67 = hand_landmarks.landmark[9]
            # Check for right hand
            if handedness.classification[0].label == "Left":
                #print("Right hand landmarks:")
                lm67 = hand_landmarks.landmark[9]
                if prev ==-1:
                    prev = lm67.y
                if prev != -1:
                    if abs(lm67.y - prev) > 0.05:
                        #print ("right", rm.y, "left",lm.y)
                        if abs(rm67.y - lm67.y) > 0.05:
                            print("SIX SEVEN")
                            last = 3
                            did67 = 20
                    prev = lm67.y

                #print(f"Landmark: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")

    #uncomment all of this if you want to see the wireframe stuff
    #if results.multi_hand_landmarks:
     # for hand_landmarks in results.multi_hand_landmarks:
      #  mp_drawing.draw_landmarks(
       #     image,
        #    hand_landmarks,
         #   mp_hands.HAND_CONNECTIONS,
          #  mp_drawing_styles.get_default_hand_landmarks_style(),
           # mp_drawing_styles.get_default_hand_connections_style())

   # if results2.detections:
    #    for detection in results2.detections:
     #       mp_drawing.draw_detection(image, detection)


    if did67 >0 and last ==3:
        image_height, image_width, _ = image.shape
        cx, cy = int(rm67.x * image_width), int(rm67.y * image_height)
        cy -= 300
        text = "6"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 6.0
        thickness = 15

        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        temp = np.zeros((text_h + 10, text_w + 10, 3), dtype=np.uint8)
        cv2.putText(temp, text, (5, text_h + 2), font, scale, (255, 0, 0), thickness)
        temp_flipped = cv2.flip(temp, 1)

        x1 = max(cx - text_w // 2, 0)
        y1 = max(cy - text_h // 2, 0)
        x2 = min(x1 + temp_flipped.shape[1], image_width)
        y2 = min(y1 + temp_flipped.shape[0], image_height)

        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        image[y1:y1 + roi_h, x1:x1 + roi_w] = cv2.addWeighted(
            roi, 1, temp_flipped[:roi_h, :roi_w], 1, 0
        )
        #other hand
        image_height, image_width, _ = image.shape
        cx, cy = int(lm67.x * image_width), int(lm67.y * image_height)
        cy -= 300

        text = "7"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 6.0
        thickness = 15

        (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
        temp = np.zeros((text_h + 10, text_w + 10, 3), dtype=np.uint8)

        cv2.putText(temp, text, (5, text_h + 2), font, scale, (255, 0, 0), thickness)
        temp_flipped = cv2.flip(temp, 1)

        x1 = max(cx - text_w // 2, 0)
        y1 = max(cy - text_h // 2, 0)
        x2 = min(x1 + temp_flipped.shape[1], image_width)
        y2 = min(y1 + temp_flipped.shape[0], image_height)

        roi = image[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]

        image[y1:y1 + roi_h, x1:x1 + roi_w] = cv2.addWeighted(
            roi, 1, temp_flipped[:roi_h, :roi_w], 1, 0
        )

    if didcry >0 and last == 1:
        if results2.detections:
            ih, iw, _ = image.shape
            for detection in results2.detections:
                keypoints = detection.location_data.relative_keypoints
                kp = keypoints[2]
                cx, cy = int(kp.x * iw), int(kp.y * ih)

                emoji = cv2.imread("crying_emoji.png", cv2.IMREAD_UNCHANGED)

                size = 250
                emoji = cv2.resize(emoji, (size, size), interpolation=cv2.INTER_AREA)

                emoji = cv2.flip(emoji, 1)

                h, w = emoji.shape[:2]

                x1 = max(cx - w // 2, 0)
                y1 = max(cy - h // 2, 0)
                x2 = min(x1 + w, iw)
                y2 = min(y1 + h, ih)

                emo_roi = emoji[:(y2 - y1), :(x2 - x1)]
                if emo_roi.size == 0:
                    continue

                bgr_roi = image[y1:y2, x1:x2]
                alpha = emo_roi[:, :, 3:4] / 255.0  # (H,W,1)
                color = emo_roi[:, :, :3]  # (H,W,3)
                bgr_roi[:] = (bgr_roi * (1 - alpha) + color * alpha).astype(np.uint8)

    if didnya >0 and last == 2:
        if results2.detections:
            ih, iw, _ = image.shape
            for detection in results2.detections:
                keypoints = detection.location_data.relative_keypoints

                kp = keypoints[4]
                cx, cy = int(kp.x * iw), int(kp.y * ih)

                cy -= 180  #height
                emoji = cv2.imread("anger.png", cv2.IMREAD_UNCHANGED)

                #size
                size = 170
                emoji = cv2.resize(emoji, (size, size), interpolation=cv2.INTER_AREA)
                emoji = cv2.flip(emoji, 1)
                h, w = emoji.shape[:2]

                x1 = max(cx - w // 2, 0)
                y1 = max(cy - h // 2, 0)
                x2 = min(x1 + w, iw)
                y2 = min(y1 + h, ih)

                emo_roi = emoji[:(y2 - y1), :(x2 - x1)]
                if emo_roi.size == 0:
                    continue

                bgr_roi = image[y1:y2, x1:x2]
                alpha = emo_roi[:, :, 3:4] / 255.0  # (H,W,1)
                color = emo_roi[:, :, :3]  # (H,W,3)
                bgr_roi[:] = (bgr_roi * (1 - alpha) + color * alpha).astype(np.uint8)


    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()