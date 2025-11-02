
## Cooper Greene 2025

import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

import face_recognition

## VERY EXPERIMENTAL RIGHT NOW :DDD

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
stopsign = cv2.CascadeClassifier('stop_sign_pjy.xml')

from datetime import datetime

cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
#width = 480
#height = 600

#cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#cap.set(cv2.CAP_PROP_FPS, 240)

while True:
    ret, frame1 = cap.read()
    if not ret:
        break

    raw = cv2.flip(frame1, 1)
    frame = cv2.flip(frame1, 1)

    # Display the original and inverted images

    # Make overlay image same size as frame
    overlay = frame.copy()

    # Example: Semi-transparent rectangle at top
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)  # black bar
    alpha = 0.4  # transparency factor

    timer = frame.copy()    
    cv2.rectangle(timer, (800, 500), (500, 600), (0,0,0), -1)

    current_process_times = os.times()

    # Blend overlay with frame
    frame = cv2.addWeighted(timer, alpha, frame, 1 - 0.4, 0)
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - 0.4, 0)

    # Add text on top
    cv2.putText(frame, "test", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Get the current local date and time
    current_datetime = datetime.now()

    # Extract only the time component
    formatted_time = current_datetime.strftime("%H:%M:%S")

    cv2.putText(frame, f"{formatted_time}", (500, 600),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Example crosshair
    h, w = frame.shape[:2]
    cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
    cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    stopsigns = stopsign.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in stopsigns:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    ##for (x,y,w,h) in face:
        ##cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,204),2)
        
    cv2.imshow("istem", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
