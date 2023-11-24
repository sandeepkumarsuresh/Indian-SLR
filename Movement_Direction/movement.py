import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

prev_hand_positions = {"Left": None, "Right": None}

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_side = handedness.classification[0].label
            hand_id = handedness.classification[0].index

            # Display hand side
            

            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            if prev_hand_positions[hand_side] is not None:
                dx, dy = (cx - prev_hand_positions[hand_side][0], cy - prev_hand_positions[hand_side][1])

                movement = ""
                if abs(dx) > abs(dy):
                    if dx > 10:
                        movement = f"{hand_side} hand moving towards the right"
                    elif dx < -10:
                        movement = f"{hand_side} hand moving towards the left"
                else:
                    if dy > 10:
                        movement = f"{hand_side} hand moving downwards"
                    elif dy < -10:
                        movement = f"{hand_side} hand moving upwards"

                if hand_side == "Left":
                    cv2.putText(frame, movement, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif hand_side == "Right":
                    cv2.putText(frame, movement, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            prev_hand_positions[hand_side] = (cx, cy)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
