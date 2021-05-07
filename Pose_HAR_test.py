# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:58:38 2021

@author: shrey
"""
import cv2
import mediapipe as mp
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.layers import Input
import PIL
import os
    
def split_sequences(sequences, n_steps, overlap):
  X = list()
  sequences_new=sequences.tolist()
 
  for i in range(0,sequences.shape[0], int(np.ceil(overlap*n_steps))):
		# find the end of this pattern
    end_ix=i+n_steps
		# check if we are beyond the dataset 
    if end_ix > sequences.shape[0]:
      break
    else:
      X.append(sequences_new[i:end_ix])
    

  X=np.asarray(X)
  Y=np.concatenate(X)
    

  return Y

tf.saved_model.LoadOptions(
    experimental_io_device=None)
model = tf.keras.models.load_model(os.getcwd())
LABELS = [    
    "JUMPING",
    "JUMPING_JACKS",
    "BOXING",
    "WAVING_2HANDS",
    "WAVING_1HAND",
    "CLAPPING_HANDS"
] 

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
idx = None 
## LIST of 18 LandMarks to extract
landmarks_list = [mp_pose.PoseLandmark.NOSE,mp_pose.PoseLandmark.LEFT_EYE,
               mp_pose.PoseLandmark.RIGHT_EYE,mp_pose.PoseLandmark.LEFT_EAR,
               mp_pose.PoseLandmark.RIGHT_EAR,mp_pose.PoseLandmark.LEFT_SHOULDER,
               mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.LEFT_ELBOW,
               mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.LEFT_WRIST,
               mp_pose.PoseLandmark.RIGHT_WRIST, 'HIP',mp_pose.PoseLandmark.LEFT_KNEE,
               mp_pose.PoseLandmark.RIGHT_KNEE,mp_pose.PoseLandmark.LEFT_HEEL,
               mp_pose.PoseLandmark.RIGHT_HEEL,mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
               mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]

Hip_tags= [mp_pose.PoseLandmark.RIGHT_HIP,mp_pose.PoseLandmark.LEFT_HIP]
# For webcam input:
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
pose=mp_pose.Pose(min_detection_confidence=0.3,min_tracking_confidence=0.3)
frame_count = 0
keypoints= []
buffer = []
while cap.isOpened():    
    success, image = cap.read()
    frame_count += 1
    if not success:

        continue
    
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = pose.process(image)
    
    
    if results.pose_landmarks:
        for landmark in landmarks_list:
            if landmark == 'HIP' :
                Hip_x = results.pose_landmarks.landmark[Hip_tags[0]].x + results.pose_landmarks.landmark[Hip_tags[1]].x
                Hip_y = results.pose_landmarks.landmark[Hip_tags[0]].y + results.pose_landmarks.landmark[Hip_tags[1]].y
                keypoints.append( Hip_x/2 * image.shape[1])
                keypoints.append( Hip_y/2 * image.shape[1])
            else:   
                keypoints.append(results.pose_landmarks.landmark[landmark].x*image.shape[1])
                #print("kp:",results.pose_landmarks.landmark[landmark].x*image.shape[1])
                keypoints.append(results.pose_landmarks.landmark[landmark].y*image.shape[0])
        
                
        #print(keypoints)
        #print(frame_count)
    
    else:
        keypoints_h = []
        for k in range(18*2):
            keypoints.append(0.0)  
            #print(keypoints)
    ## Referesh every 32 frames ##
    if frame_count == 32:
        buffer = np.asarray(keypoints)
        #print(buffer[:36])
        #break
        keypoints=[]
        frame_count = 0
        
        ##obtain vector of keypoints
        X = np.array(np.split(buffer,32))
        X=StandardScaler().fit_transform(X)
        X=X.reshape(1,32,36)
        Z=list()
        for i in range(X.shape[0]):
            Z.append(split_sequences(X[i],32,0.5))
        X=np.array(Z)       
        print(X.shape)
        
        Y=model.predict(X)
        y = Y.tolist()
        idx = y[0].index(max(y[0]))
    
    if idx == None :
        continue
    image.flags.writeable = True
    font = cv2.FONT_HERSHEY_SIMPLEX
    orgin = (int(image.shape[0]/2), int(image.shape[1]*0.7))  
    fontScale = 1
    color = (255, 255, 255) 
    thickness = 2
    image = cv2.putText(image, LABELS[idx], orgin, font,  
               fontScale, color, thickness, cv2.LINE_AA) 

    # #print(Y.shape)
        
    # Draw the pose annotation on the image.
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
pose.close()