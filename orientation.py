import cv2
import mediapipe as mp
import math

def hand_tracking_from_video(video_path='happy_1.mp4'):
    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    # Initialize variables to store reference points
    left_hand_reference = None
    right_hand_reference = None

    # Read video using OpenCV
    cap = cv2.VideoCapture(video_path)

    # Set the desired window size
    new_width, new_height = 640, 480

    # Create a resizable window
    cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Tracking', new_width, new_height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # Check if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

                # Convert keypoints to pixel coordinates
                h, w, _ = frame.shape
                wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
                middle_finger_tip_x, middle_finger_tip_y = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)

                if wrist_x < w / 2:
                    hand_side = 'Right Hand'
                    wrist_text = 'Right Hand Wrist'
                    tip_text = 'Right Hand Tip'

                    # Calculate vector from wrist to tip
                    wrist_to_tip = [middle_finger_tip_x - wrist_x, middle_finger_tip_y - wrist_y]

                    # Calculate vector from wrist to reference (adding 20 to x-axis)
                    right_hand_reference = [wrist_x + 20, wrist_y]

                else:
                    hand_side = 'Left Hand'
                    wrist_text = 'Left Hand Wrist'
                    tip_text = 'Left Hand Tip'

                    # Calculate vector from wrist to tip
                    wrist_to_tip = [middle_finger_tip_x - wrist_x, middle_finger_tip_y - wrist_y]

                    # Calculate vector from wrist to reference (adding 20 to x-axis)
                    left_hand_reference = [wrist_x + 20, wrist_y]

                # Draw points and text on the frame
                cv2.circle(frame, (wrist_x, wrist_y), 10, (0, 255, 0), -1)
                cv2.circle(frame, (middle_finger_tip_x, middle_finger_tip_y), 10, (0, 0, 255), -1)
                cv2.putText(frame, wrist_text, (wrist_x, wrist_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, tip_text, (middle_finger_tip_x, middle_finger_tip_y - 20), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.line(frame, (wrist_x, wrist_y), (middle_finger_tip_x, middle_finger_tip_y), (255, 0, 0), 2)

            if left_hand_reference is not None:
                # Draw left hand reference point
                cv2.circle(frame, (int(left_hand_reference[0]), int(left_hand_reference[1])), 10, (255, 0, 0), -1)

                # Calculate vector from wrist to reference
                wrist_to_reference = [left_hand_reference[0] - wrist_x, left_hand_reference[1] - wrist_y]

                # Calculate angle between vectors
                angle_rad = math.atan2(wrist_to_tip[1], wrist_to_tip[0]) - math.atan2(wrist_to_reference[1], wrist_to_reference[0])
                angle_deg = math.degrees(angle_rad)

                # Ensure the angle is positive (between 0 and 360 degrees)
                angle_deg = (360 - angle_deg) % 360

                # Display the angle
                cv2.putText(frame, f'Left Hand Angle: {angle_deg:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Set threshold values for orientation
                vertical_up_threshold = (45, 135)
                horizontal_rightt_threshold = (135, 225)
                vertical_down_threshold = (225, 315)
                horizontal_leftt_threshold = (315, 360)

                # Determine orientation based on the angle
                if vertical_up_threshold[0] <= angle_deg <= vertical_up_threshold[1]:
                    orientation_text = 'Vertical Up'
                elif horizontal_rightt_threshold[0] <= angle_deg <= horizontal_rightt_threshold[1]:
                    orientation_text = 'Horizontal Right'
                elif vertical_down_threshold[0] <= angle_deg <= vertical_down_threshold[1]:
                    orientation_text = 'Vertical Down'
                elif horizontal_leftt_threshold[0] <= angle_deg <= horizontal_leftt_threshold[1]:
                    orientation_text = 'Horizontal Left'
                else:
                    orientation_text = 'Undefined'

                # Display the orientation
                cv2.putText(frame, f'Orientation: {orientation_text}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            if right_hand_reference is not None:
                # Draw right hand reference point
                cv2.circle(frame, (int(right_hand_reference[0]), int(right_hand_reference[1])), 10, (255, 0, 0), -1)

                # Calculate vector from wrist to reference
                wrist_to_reference = [right_hand_reference[0] - wrist_x, right_hand_reference[1] - wrist_y]

                # Calculate angle between vectors
                angle_rad = math.atan2(wrist_to_tip[1], wrist_to_tip[0]) - math.atan2(wrist_to_reference[1], wrist_to_reference[0])
                angle_deg = math.degrees(angle_rad)

                # Ensure the angle is positive (between 0 and 360 degrees)
                angle_deg = (360 - angle_deg) % 360

                # Display the angle
                cv2.putText(frame, f'Right Hand Angle: {angle_deg:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

                # Set threshold values for orientation
                vertical_up_threshold = (45, 135)
                horizontal_rightt_threshold = (135, 225)
                vertical_down_threshold = (225, 315)
                horizontal_leftt_threshold = (315, 360)

                # Determine orientation based on the angle
                if vertical_up_threshold[0] <= angle_deg <= vertical_up_threshold[1]:
                    orientation_text = 'Vertical Up'
                elif horizontal_rightt_threshold[0] <= angle_deg <= horizontal_rightt_threshold[1]:
                    orientation_text = 'Horizontal Right'
                elif vertical_down_threshold[0] <= angle_deg <= vertical_down_threshold[1]:
                    orientation_text = 'Vertical Down'
                elif horizontal_leftt_threshold[0] <= angle_deg <= horizontal_leftt_threshold[1]:
                    orientation_text = 'Horizontal Left'
                else:
                    orientation_text = 'Undefined'

                # Display the orientation
                cv2.putText(frame, f'Orientation: {orientation_text}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the frame in the resizable window
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Call the function to run hand tracking from the specified video
hand_tracking_from_video()
