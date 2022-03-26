#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  7 20:16:27 2022

@author: danielcoelho
"""

import face_recognition
import cv2

# Getting the webcam #0 (the default one, 1, 2, 3, etc means more the one camera)
webcam_video_stream = cv2.VideoCapture(0)

# Initialize the array variable to hold all face locations in the frame
all_face_locations = []

while True:
    # Getting current frame
    ret, current_frame = webcam_video_stream.read()
    
    # resizing the current frame to 1/4 size to process faster
    current_frame_small = cv2.resize(current_frame, (0,0), fx=0.25, fy=0.25)
    
    # Detect all faces in the image -- model = 'cnn' can be used but takes long, but more accurate
    all_face_location = face_recognition.face_locations(current_frame_small, number_of_times_to_upsample = 2, model = 'hog')
    
    # Loopiong through to find face location
    for index, current_face_location in enumerate(all_face_location):
        
        # Spliting the tuple for each position values
        top_pos, right_pos, bottom_pos, left_pos = current_face_location
        
        # Have to times everything by 4 due to sizing
        top_pos = top_pos*4
        right_pos = right_pos*4
        bottom_pos = bottom_pos*4
        left_pos = left_pos*4
        
        # Printing the location of current face
        print('Found face {} at top: {}, right: {}, bottom: {}, left: {}'.format(index +1,top_pos, right_pos, bottom_pos, left_pos))
        
        # Drawing rectanle around each face location in the main video 
        cv2.rectangle(current_frame, (left_pos, top_pos), (right_pos, bottom_pos), (0,0,255), 2)
        
    #Showing the current face with rectangle drawn 
    cv2.imshow("Webcam Video", current_frame)
        
    # To break the while loop (32bit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
# Releasing the stream and cam // closing all opencv windows opened
webcam_video_stream.release()
cv2.destroyAllWindows()
