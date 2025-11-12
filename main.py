
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
import textwrap

print("iSTEM AT3 2025 - Cooper Greene")

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

def get_speed_zone(lat, lon, gdf, tolerance=10):
    # ensure WGS84 coords
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    point = Point(lon, lat)

    # spatial index filter first
    idx = list(gdf.sindex.intersection(point.bounds))
    matches = gdf.iloc[idx]

    # compute distance to each line
    matches["dist"] = matches.geometry.distance(point)

    # pick closest road segment
    nearest = matches.loc[matches["dist"].idxmin()]

    # if too far, assume not a road match
    # ~0.0001 degrees ≈ 11 m
    if nearest["dist"] > 0.0001:
        return None

    return nearest.to_dict()

SPEED_ZONE = None

def tfNSW_check(SPEED_ZONE):
    if STOP_THREADS:
        print("\nexit()")
        return
    # schedule the next call first
    lat, lon = get_current_location()

    #lat = -34.063735
    #lon = 150.731770

    print(f"Your current location: {lat}, {lon}\n")
    print("check")
    hazards = get_live_hazards("incident", "open")
    print(f"Fetched {len(hazards.get('features', []))} live incidents.")

    nearby_hazards = filter_nearby_hazards(hazards, lat, lon)
    for h in nearby_hazards:
        print(f"{h['properties']['displayName']} - {h['distance_km']} km away")

    zone = get_speed_zone(lat, lon, gdf)
    if zone:
        print(f"Current speed limit: {zone['Speed']} km/h")
        SPEED_ZONE = zone['Speed']
        print(SPEED_ZONE)
    else:
        print("Could not determine speed zone.")

    threading.Timer(15, tfNSW_check, args=[SPEED_ZONE]).start()
    return SPEED_ZONE,hazards,nearby_hazards

STOP_THREADS = False
API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJrNmdvdWJoOFpBQ19jbWpYTlNBemFKemR1NUJjT2k1dWFoQjFWZ2haM0k4IiwiaWF0IjoxNzYyNTEyNTA0fQ.k2VZRUPA-WkdMyjYpYYhYl0lqc0SDxUT0UKxEton5wA"
BASE_URL = "https://api.transport.nsw.gov.au/v1/live/hazards"

print("\ninit...")

gdf = gpd.read_parquet("speed_zones.parquet")
print(gdf.crs)
# Fix CRS
if gdf.crs is None:
    gdf.set_crs("EPSG:4326", inplace=True)
if gdf.crs.to_string() != "EPSG:4326":
    gdf = gdf.to_crs(epsg=4326)
print(gdf.total_bounds)

SPEED_ZONE,hazards,nearby_hazards = tfNSW_check(SPEED_ZONE)

print("Starting lfCamera")

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
stopsign = cv2.CascadeClassifier('stop_sign_pjy.xml')

if sys.platform == 'darwin':
    cap = cv2.VideoCapture(1, cv2.CAP_AVFOUNDATION)
    if not cap:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
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
    
    alpha = 0.4  # transparency factor

    timer = frame.copy()
    cv2.rectangle(timer, (900, 50), (1200, 900), (0,0,0), -1)
    #cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift)

    current_process_times = os.times()

    # Blend overlay with frame
    frame = cv2.addWeighted(timer, alpha, frame, 1 - 0.4, 0)

    # Get the current local date and time
    current_datetime = datetime.now()

    # Extract only the time component
    formatted_time = current_datetime.strftime("%H:%M:%S")

    cv2.putText(frame, f"{formatted_time}", (1050, 850),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #print(frame.shape)

    height, width, channel = frame.shape

    text_img = np.ones((height, width))
    #print(text_img.shape)
    font = cv2.FONT_HERSHEY_SIMPLEX
    wrapped_text = []
    for h in nearby_hazards:
        #print(wrapped_text)
        n = textwrap.wrap(f"{h['properties']['displayName']} - {h['distance_km']} km away | ", width=15)
        wrapped_text.extend(n)

    font_size = 1
    font_thickness = 2

    for i, line in enumerate(wrapped_text):
        textsize = cv2.getTextSize(line, font, font_size, font_thickness)[0]
        #print(textsize)

        gap = textsize[1] + 10

        y = int((547 + textsize[1]) / 2) + i * gap
        x = 900
        #print(x, y)

        cv2.putText(frame, line, (x, y), font,
                    font_size, 
                    (255,255,255), 
                    font_thickness, 
                    lineType = cv2.LINE_AA)

    
    if SPEED_ZONE:
        speedzone = SPEED_ZONE.replace(" km/h","")
    else:
        #print("No speed zone!")
        speedzone = "???"

    cv2.circle(frame,(1050,150), 63, (0,0,255), 7)
    cv2.circle(frame, center=(1050, 150), radius=63, color=(255, 255, 255), thickness=-1)  # Filled Blue Circle
    cv2.putText(frame, f"{speedzone}", (1030, 155),
                cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 0, 0), 2)


    # Example crosshair
    h, w = frame.shape[:2]
    cv2.line(frame, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 0), 2)
    cv2.line(frame, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 0), 2)

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2)
    stopsigns = stopsign.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in stopsigns:
        #playsound('sod.mp3', block=False)
        print("Stop sign!")
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    for (x,y,w,h) in face:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,204),2)
    
    #cv2.putText(image, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

    #cv2.circle(frame, (447, 63), radius, color, thickness)
        
    cv2.imshow("NEO lfCamera", frame)
    #print(SPEED_ZONE)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


STOP_THREADS = True
print("Exiting threads")
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
