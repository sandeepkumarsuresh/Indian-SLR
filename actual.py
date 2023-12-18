import cv2
import numpy as np
import mediapipe as mp

# Initialize Mediapipe Hand module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize Pose module from Mediapipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capture video stream
cap = cv2.VideoCapture("C:/Users/RP/Downloads/help.mp4")


def fingertip(hands_results):
    fingertips_joined = False
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Extract fingertip coordinates
            fingertips = [hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
                           hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]]

            # Define a threshold distance for fingertips to be considered joined
            threshold_distance = 0.05  # Adjust as needed

            # Check if fingertips are joined
            fingertips_joined = all(cv2.norm((fingertips[i - 1].x, fingertips[i - 1].y),
                                            (fingertips[i].x, fingertips[i].y)) < threshold_distance
                                    for i in range(1, len(fingertips)))

            # Draw the landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Print whether fingertips are joined or not
            print("Fingertips joined:", fingertips_joined)

def Detect_fist(hands_results):

    fist_detected = False
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            # Extract hand landmarks
            index_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            indexmid_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * frame.shape[0])
            bottom_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # Check if the hand is in a fist position
            fist_detected = (index_y < bottom_y) and (index_y > indexmid_y)

            # # Draw bounding box for the fist
            # indexX = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            # indexY = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])
            # pinkyX = int(hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * frame.shape[1])
            # handBottomY = int(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * frame.shape[0])

            # cv2.rectangle(frame, (indexX, indexY), (pinkyX, handBottomY), (0, 0, 255), 2)
            

            # Draw the landmarks on the frame
            # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            print("Fist Detected:", fist_detected)

def shoulder(results):

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
                print("Left hand is near the face")
            else:
                print("Left hand is near the chest")

            if right_hand.y < expected_y_right + (threshold * (right_shoulder.y - left_shoulder.y)):
                print("Right hand is near the face")
            else:
                print("Right hand is near the chest")

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def extract_fingertip_and_wrist(image):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_drawing = mp.solutions.drawing_utils

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and get the hand landmarks
    results = hands.process(rgb_image)

    # List to store fingertip, baseline, handness, and normalized distance for each hand
    fingertips_and_wrists = []

    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract fingertip points for each finger
            fingertip_points = [(int(hand_landmarks.landmark[i].x * image.shape[1]), int(hand_landmarks.landmark[i].y * image.shape[0])) for i in range(4, 21, 4)]

            # Extract baseline points (e.g., index finger base and wrist base)
            baseline_point1 = (int(hand_landmarks.landmark[5].x * image.shape[1]), int(hand_landmarks.landmark[5].y * image.shape[0]))
            baseline_point2 = (int(hand_landmarks.landmark[0].x * image.shape[1]), int(hand_landmarks.landmark[0].y * image.shape[0]))

            # Calculate Euclidean distance between baseline and wrist
            distance_baseline_wrist = calculate_distance(baseline_point1, baseline_point2)

            # Calculate normalized distance for each finger
            normalized_distances = [distance_baseline_wrist / calculate_distance(fingertip_point, baseline_point2) for fingertip_point in fingertip_points]

            # Determine hand type based on landmark[0].x
            hand_type = "Left" if hand_landmarks.landmark[0].x < 0.5 else "Right"

            # Draw landmarks on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Define finger names in a standard order (thumb, index, middle, ring, pinky)
            finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

            # Determine finger status for each finger
            finger_statuses = [{"Name": finger_names[i], "Status": "Open" if 0 < dist <= 0.8 else "Closed"} for i, dist in enumerate(normalized_distances)]

            # Store fingertip, baseline, handness, finger names, and finger statuses in the list
            fingertips_and_wrists.append({
                "hand_type": hand_type,
                "finger_statuses": finger_statuses
            })

    return image, fingertips_and_wrists




while cap.isOpened():
    # Read each frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to get hand landmarks
    hands_results = hands.process(rgb_frame)

    
    # Detect joined fingertips
    fingertip(hands_results)



    # Detect a fist
    Detect_fist(hands_results)


    # # Process the frame with Mediapipe
    results = pose.process(rgb_frame)

    # # Get shoulder landmarks

    shoulder(results)

    frame, fingertips_and_wrists = extract_fingertip_and_wrist(frame)

    for i, points in enumerate(fingertips_and_wrists):
        print(f"Hand {i + 1} - Type: {points['hand_type']}, Finger Statuses: {points['finger_statuses']}")


    #     # Display landmarks on shoulders
    #     cv2.circle(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])), 5, (255, 0, 0), -1)
    #     cv2.circle(frame, (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), 5, (255, 0, 0), -1)

    #     # Display the line connecting the shoulders
    #     cv2.line(frame, (int(left_shoulder.x * frame.shape[1]), int(left_shoulder.y * frame.shape[0])),
    #              (int(right_shoulder.x * frame.shape[1]), int(right_shoulder.y * frame.shape[0])), (0, 255, 0), 2)
        
    # cv2.imshow('Hand Tracking and Pose Estimation', frame)


    # # Press 'Esc' to exit
    # if cv2.waitKey(1) & 0xFF == 27:
    #     break

# Release resources
cap.release()
cv2.destroyAllWindows()
