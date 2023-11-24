import cv2
import mediapipe as mp

# Initialize Pose module from Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capture video stream
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Read each frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe
    results = pose.process(rgb_frame)

    # Get shoulder landmarks
    if results.pose_landmarks:
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]

        # Calculate the equation of the line: y = mx + b
        m = (right_shoulder.y - left_shoulder.y) / (right_shoulder.x - left_shoulder.x)
        b = left_shoulder.y - m * left_shoulder.x

        # Get hand landmarks
        if results.pose_landmarks:
            left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
            right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]

            # Calculate the expected y-coordinate on the line for each hand
            expected_y_left = m * left_hand.x + b
            expected_y_right = m * right_hand.x + b

            # Determine the threshold for considering the hand above the line (adjust as needed)
            threshold = 3 / 4

            # Check if a portion of the hand is above the line
            if left_hand.y < expected_y_left + (threshold * (right_shoulder.y - left_shoulder.y)):
                cv2.putText(frame, "Left hand is near the face", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Left hand is near the chest", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            if right_hand.y < expected_y_right + (threshold * (right_shoulder.y - left_shoulder.y)):
                cv2.putText(frame, "Right hand is near the face", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "Right hand is near the chest", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display landmarks on shoulders
        cv2.circle(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])), 5, (255, 0, 0), -1)
        cv2.circle(frame, (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), 5, (255, 0, 0), -1)

        # Display the line connecting the shoulders
        cv2.line(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])),
                 (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), (0, 255, 0), 2)

    # Display the results
    cv2.imshow("Pose Estimation", frame)

    # Press 'Esc' to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
