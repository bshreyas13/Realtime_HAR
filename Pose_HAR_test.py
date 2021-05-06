# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:58:38 2021

@author: shrey
"""
import cv2
import mediapipe as mp
import numpy as np
    
def split_sequences(sequences, n_steps):
  X = list()
  sequences_new=sequences.tolist()
 
  for i in range(0,sequences.shape[0],10):
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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

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
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
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
                keypoints.append(results.pose_landmarks.landmark[landmark].y*image.shape[0])
        #print(keypoints_frame)
        #print(len(keypoints))
        #break
    ## Referesh every 32 frames ##
    if frame_count == 32:
        buffer = np.asarray(keypoints)
        keypoints=[]
        frame_count=0
        
        ##obtain vector of keypoints
        X = np.array(np.split(buffer,32))
        X = X.reshape(1,X.shape[0],X.shape[1])
        #print(X.shape)
        
    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('MediaPipe Pose', image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
      break
cap.release()
cv2.destroyAllWindows()
pose.close()