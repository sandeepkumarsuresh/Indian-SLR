import cv2
import mediapipe as mp
import numpy as np

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

def main():
    # Open the webcam
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Read a frame from the webcam
        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame")
            break

        # Detect hands, draw landmarks, and extract fingertip, baseline, and calculate normalized distance
        frame, fingertips_and_wrists = extract_fingertip_and_wrist(frame)

        # Display the output
        cv2.imshow("Hand Detection", frame)

        # Print information for each hand
        for i, points in enumerate(fingertips_and_wrists):
            print(f"Hand {i + 1} - Type: {points['hand_type']}, Finger Statuses: {points['finger_statuses']}")

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
