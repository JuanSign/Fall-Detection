import cv2
import mediapipe as mp

class MediaPipe_Pipeline:
    def __init__(self, pose, mp_drawing):
        self.pose = pose
        self.mp_drawing = mp_drawing
    
    @staticmethod
    def process_image(path, output=False):
        image = cv2.imread(path)
        
        image_height, image_width, _ = image.shape
        
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        results = pose.process(rgb_img)        

        resultMap = {}

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmark_name = mp_pose.PoseLandmark(idx).name

                x = int(landmark.x * image_width)  
                y = int(landmark.y * image_height) 
                
                confidence = landmark.visibility

                if confidence >= 0.88:
                    resultMap[landmark_name + '_X'] = x
                    resultMap[landmark_name + '_Y'] = y 

                if output:
                    print(f"{landmark_name}: ({x}, {y}), Confidence: {confidence:.2f}")

        if output:
            cv2.imshow("Pose Estimation", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return resultMap

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # Example usage:
# pipeline = MediaPipe_Pipeline(pose, mp_drawing)
# result = pipeline.process_image(r'data\dataset\train\subject-1\fall\1_backward_falls\frame001.jpg', False)
# print(result)