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
                if (abs(rm1.z-rm2.z) <0.008):
                    if abs(rm2.z-rm3.z) < 0.008:
                        if abs(rm3.z-rm4.z) < 0.008:
                            if (abs(lm1.z - lm2.z) < 0.008):
                                if abs(lm2.z - lm3.z) < 0.008:
                                    if abs(lm3.z - lm4.z) < 0.008:
                                        #print("similar")
                                        if prevr == -1:
                                            prevr = rm2.x
                                            prevl = lm2.x
                                        else:
                                            #print(prevr,rm2.x)
                                            if abs(prevr - rm2.x) > 0.008:
                                                if abs(prevl - lm2.x) > 0.008:
                                                    print("Cry cry")
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
                if (abs(rm1.z - rm2.z) < 0.008):
                    if abs(rm2.z - rm3.z) < 0.008:
                        if abs(rm3.z - rm4.z) < 0.008:
                            if (abs(lm1.z - lm2.z) < 0.008):
                                if abs(lm2.z - lm3.z) < 0.008:
                                    if abs(lm3.z - lm4.z) < 0.008:
                                        # print("similar")
                                        if prevr == -1:
                                            prevr2 = rm2.x
                                            prevl2 = lm2.x
                                        else:
                                            # print(prevr,rm2.x)
                                            if abs(prevr2 - rm2.x) > 0.008:
                                                if abs(prevl2 - lm2.x) > 0.008:
                                                    print("nya nya")
                                                    prevr2 = rm2.x
                                                    prevl2 = lm2.x




    ##67 stuff lol
    did67 = -1
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness):
            did67 = did67 - 1
            if handedness.classification[0].label == "Right":
                rm = hand_landmarks.landmark[9]
            # Check for right hand
            if handedness.classification[0].label == "Left":
                #print("Right hand landmarks:")
                lm = hand_landmarks.landmark[9]
                if prev ==-1:
                    prev = lm.y
                if prev != -1:
                    if abs(lm.y - prev) > 0.05:
                        #print ("right", rm.y, "left",lm.y)
                        if abs(rm.y - lm.y) > 0.05:
                            print("SIX SEVEN")
                            did67 = 3
                        #exit()
                    else:
                        if did67 >0:
                            print("SIX SEVEN")
                        #else :
                           #print("no")
                    prev = lm.y

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