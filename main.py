
## Cooper Greene 2025

# Imports
import cv2
import os
import sys
import numpy as np
from playsound3 import playsound
from datetime import datetime
import pyarrow
import geocoder
import requests
from geopy.distance import distance
import json
from shapely.geometry import shape, Point
import geopandas as gpd
import sched, time
import threading

def get_current_location():
    g = geocoder.ip('me')
    if g.ok:
        return g.latlng  # [latitude, longitude]
    else:
        raise Exception("Unable to determine current location.")
def get_live_hazards(hazard_type="incident", status="open"):
    url = f"{BASE_URL}/{hazard_type}/{status}"
    headers = {"Authorization": f"apikey {API_KEY}"}
    r = requests.get(url, headers=headers,verify=False)
    r.raise_for_status()
    return r.json()

def filter_nearby_hazards(hazards, lat, lon, radius_km=5):
    nearby = []
    for feature in hazards.get("features", []):
        coords = feature["geometry"]["coordinates"]
        # GeoJSON coords are [lon, lat]
        dist = distance((lat, lon), (coords[1], coords[0])).km
        if dist <= radius_km:
            feature["distance_km"] = round(dist, 2)
            nearby.append(feature)
    return nearby

def get_speed_zone(lat, lon, gdf):
    point = Point(lon, lat)
    matches = gdf[gdf.contains(point)]
    if not matches.empty:
        return matches.iloc[0].to_dict()
    return None

def tfNSW_check(): 
    # schedule the next call first
    print("check")
    hazards = get_live_hazards("incident", "open")
    print(f"Fetched {len(hazards.get('features', []))} live incidents.")

    nearby_hazards = filter_nearby_hazards(hazards, lat, lon)
    for h in nearby_hazards:
        print(f"{h['properties']['headline']} - {h['distance_km']} km away")

    zone = get_speed_zone(lat, lon, gdf)
    if zone:
        print(f"Current speed limit: {zone['SPEED_LIMIT']} km/h")
    else:
        print("Could not determine speed zone.")

    threading.Timer(35, tfNSW_check).start()

lat, lon = get_current_location()
print(f"Your current location: {lat}, {lon}")

API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJrNmdvdWJoOFpBQ19jbWpYTlNBemFKemR1NUJjT2k1dWFoQjFWZ2haM0k4IiwiaWF0IjoxNzYyNTEyNTA0fQ.k2VZRUPA-WkdMyjYpYYhYl0lqc0SDxUT0UKxEton5wA"
BASE_URL = "https://api.transport.nsw.gov.au/v1/live/hazards"
print("\ninit...")

gdf = gpd.read_parquet("speed_zones.parquet")
tfNSW_check()

print("ok doen that shit")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
stopsign = cv2.CascadeClassifier('stop_sign_pjy.xml')

if sys.platform == 'darwin':
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)
cap.set(cv2.CAP_PROP_FPS, 240) # Attempt to set FPS
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
        playsound('sod.mp3', block=False)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    ##for (x,y,w,h) in face:
        ##cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,204),2)
        
    cv2.imshow("istem", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
exit()
