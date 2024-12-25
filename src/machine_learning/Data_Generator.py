import pandas as pd
from ..pose_estimation.MediaPipe import MediaPipe_Pipeline, mp
import os
import cv2
import csv

columns = [
    'id',
    'path',
    'label',
    'NOSE_X' , 'NOSE_Y' ,
    'LEFT_EYE_INNER_X' , 'LEFT_EYE_INNER_Y' ,
    'LEFT_EYE_X' , 'LEFT_EYE_Y' ,
    'LEFT_EYE_OUTER_X' , 'LEFT_EYE_OUTER_Y' ,
    'RIGHT_EYE_INNER_X' , 'RIGHT_EYE_INNER_Y' ,
    'RIGHT_EYE_X' , 'RIGHT_EYE_Y' ,
    'RIGHT_EYE_OUTER_X' , 'RIGHT_EYE_OUTER_Y' ,
    'LEFT_EAR_X' , 'LEFT_EAR_Y' ,
    'RIGHT_EAR_X' , 'RIGHT_EAR_Y' ,
    'MOUTH_LEFT_X' , 'MOUTH_LEFT_Y' ,
    'MOUTH_RIGHT_X' , 'MOUTH_RIGHT_Y' ,
    'LEFT_SHOULDER_X' , 'LEFT_SHOULDER_Y' ,
    'RIGHT_SHOULDER_X' , 'RIGHT_SHOULDER_Y' ,
    'LEFT_ELBOW_X' , 'LEFT_ELBOW_Y' ,
    'RIGHT_ELBOW_X' , 'RIGHT_ELBOW_Y' ,
    'LEFT_WRIST_X' , 'LEFT_WRIST_Y' ,
    'RIGHT_WRIST_X' , 'RIGHT_WRIST_Y' ,
    'LEFT_PINKY_X' , 'LEFT_PINKY_Y' ,
    'RIGHT_PINKY_X' , 'RIGHT_PINKY_Y' ,
    'LEFT_INDEX_X' , 'LEFT_INDEX_Y' ,
    'RIGHT_INDEX_X' , 'RIGHT_INDEX_Y' ,
    'LEFT_THUMB_X' , 'LEFT_THUMB_Y' ,
    'RIGHT_THUMB_X' , 'RIGHT_THUMB_Y' ,
    'LEFT_HIP_X' , 'LEFT_HIP_Y' ,
    'RIGHT_HIP_X' , 'RIGHT_HIP_Y' ,
    'LEFT_KNEE_X' , 'LEFT_KNEE_Y' ,
    'RIGHT_KNEE_X' , 'RIGHT_KNEE_Y' ,
    'LEFT_ANKLE_X' , 'LEFT_ANKLE_Y' ,
    'RIGHT_ANKLE_X' , 'RIGHT_ANKLE_Y' ,
    'LEFT_HEEL_X' , 'LEFT_HEEL_Y' ,
    'RIGHT_HEEL_X' , 'RIGHT_HEEL_Y' ,
    'LEFT_FOOT_INDEX_X' , 'LEFT_FOOT_INDEX_Y' ,
    'RIGHT_FOOT_INDEX_X' , 'RIGHT_FOOT_INDEX_Y'
]
header = [columns]
data = pd.DataFrame(columns)

root = r'data/dataset/train'
subject = ['subject-1', 'subject-2', 'subject-3', 'subject-4']
fallType = ['_backward_falls', '_forward_falls', '_left_falls', '_right_falls', '_sitting_falls', '_standing_falls']
nonFallType = ['_jumping', '_laying', '_picking', '_squat', '_stretching', '_walking']

# Initialize MediaPipe Pose and Drawing modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Example usage:
pipeline = MediaPipe_Pipeline(pose, mp_drawing)

for s in range(4):
    for f in range(2):
        with open(f"{subject[s]}_{'fall' if f == 0 else 'non_fall'}.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(header)
            id = 0
            for t in range(6):
                if f == 0:
                    path = root + '/' + subject[s] + '/' + 'fall/' + subject[s][-1] + fallType[t]
                if f == 1:
                    path = root + '/' + subject[s] + '/' + 'non_fall/' + subject[s][-1] + nonFallType[t]

                for img_name in os.listdir(path):
                    data = []
                    if img_name.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(path, img_name)
                        # img = cv2.imread(img_path)
                        result = pipeline.process_image(img_path, False)
                        img_data = []
                        if result is not None:
                            id+=1
                            img_data.append(id)
                            img_data.append(img_path)
                            label = 1 if f == 0 else 0
                            img_data.append(label)
                            for idx_col in range (3, len(columns)):
                                if columns[idx_col] in result and result[columns[idx_col]] is not None:
                                    img_data.append(result[columns[idx_col]])
                                else:
                                    img_data.append(None)
                        data.append(img_data)
                    writer.writerows(data)