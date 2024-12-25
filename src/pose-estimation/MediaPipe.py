import cv2
import mediapipe as mp

# definition
class MediaPipe_Pipeline:
    def __init__(self, pose, mp_drawing):
        self.pose = pose
        self.mp_drawing = mp_drawing
    
    @staticmethod
    def process_image(path):
        image = cv2.imread(path)
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_img)        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Pose Estimation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# example usage
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=2 
)
mp_drawing = mp.solutions.drawing_utils

pipeline = MediaPipe_Pipeline(pose, mp_drawing)
pipeline.process_image(r'PATH')