import pandas as pd
import os

data = pd.DataFrame(columns=[
    'id',
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
])

root = r'data/train'
subject = ['subject-1', 'subject-2', 'subject-3', 'subject-4']
fallType = ['_backward_falls', '_forward_falls', '_left_falls', '_right_falls', '_sitting_falls', '_standing_fall']
nonFallType = ['_jumping', '_laying', '_picking', '_squat', '_stretching', '_walking']

for s in range(4):
    for f in range(2):
        for t in range(6):
            if f == 0:
                path = root + '/' + subject[s] + '/' + 'fall/' + subject[s][-1] + fallType[t]
            if f == 1:
                path = root + '/' + subject[s] + '/' + 'non_fall/' + subject[s][-1] + nonFallType[t]