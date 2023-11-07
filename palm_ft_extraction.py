import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture('Cropped_Signs/hello.mp4')
# cap = cv2.VideoCapture(0)


if __name__ == '__main__':                                                                                        

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            # ret , frame = video.read
            image = cv2.resize(frame, (500, 500))
            image = cv2.flip(image, -1)
            
            # Recolor image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)




            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Get coordinates


                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].z]
                right_thumb = [landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value].z]
                right_index = [landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value].z]
                right_pinky = [landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].x , landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].y , landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value].z]


                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x , landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y , landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].z]
                left_thumb = [landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].x , landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].y , landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value].z]
                left_index = [landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].x , landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].y , landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value].z]
                left_pinky = [landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].x , landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].y , landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value].z]

                print('right_thumb',right_thumb)
                print('right index', right_index)
                print('right PINKY', right_pinky)


            
            
                # TO DO
                # if Statement:(Right thumb or right index) x co-ordinate greater x co-ordinate of wrist, perform something
                # if ((right_thumb[0])<right_index[0]):
                
                if ((right_thumb[1])<right_pinky[1]):
                    
                    # Descriptors['hands_involved']='Both hands'
                    print("Palm facing towards chest")
                
                else:
                    # Descriptors['hands_involved']='Right Hand'
                    print('Palm facing towards camera')
                    
                # Visualize angle
                # cv2.putText(image, str(right_angle), 
                #                tuple(np.multiply(right_elbow, [640, 480]).astype(int)), 
                #                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                #                     )
                # cv2.putText(image, str(left_angle), 
                #     tuple(np.multiply(left_elbow, [640, 480]).astype(int)), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                #         )
            except:
                pass
            
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
