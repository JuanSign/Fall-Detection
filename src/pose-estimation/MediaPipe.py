import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=2 
)

mp_drawing = mp.solutions.drawing_utils

path = r'data\train\subject-1\fall\1_sitting_falls\frame008.jpg'
image = cv2.imread(path)

image_height, image_width, _ = image.shape

rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = pose.process(rgb_image)

if results.pose_landmarks:
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

cv2.imshow("Pose Estimation", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
