import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

cap = cv2.VideoCapture(0)

buffer_size = 5  # Adjust the buffer size based on your preferences
prev_hand_positions = {"Left": [], "Right": []}
movement_direction = {"Left": 0, "Right": 0}

def update_buffer(buffer, value):
    buffer.append(value)
    if len(buffer) > buffer_size:
        buffer.pop(0)

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_side = handedness.classification[0].label
            hand_id = handedness.classification[0].index

            # Use the z-coordinate of the WRIST landmark for depth information
            landmark_z = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cz = int(landmark.z * w)  # Adjust for visualization, z-coordinate is scaled to width
                cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                cv2.putText(frame, str(idx), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Update the buffer with the current hand position
            update_buffer(prev_hand_positions[hand_side], (cx, cy, cz))

            # Calculate the average hand position from the buffer
            avg_hand_position = np.mean(prev_hand_positions[hand_side], axis=0)

            if len(prev_hand_positions[hand_side]) == buffer_size:
                dz = cz - avg_hand_position[2]

                # Update the movement direction
                movement_direction[hand_side] += dz

                movement = ""
                if movement_direction[hand_side] > 10:
                    movement = f"{hand_side} hand moving backward"
                elif movement_direction[hand_side] < -10:
                    movement = f"{hand_side} hand moving forward"

                # Print status on separate lines
                if hand_side == "Left":
                    cv2.putText(frame, movement, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif hand_side == "Right":
                    cv2.putText(frame, movement, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('Hand Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
