import os
import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials,db, storage
from datetime import datetime

cred = credentials.Certificate("face recog.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://faceattendance-7e387-default-rtdb.firebaseio.com/',
    'storageBucket':'faceattendance-7e387.appspot.com'
})
bucket = storage.bucket()
ref = db.reference("Students")

cfg= "darknet\cfg\yolov3-tiny.cfg"
weight= "yolov3-tiny.weights"
net = cv2.dnn.readNet(weight,cfg)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

def detect_faces_yolo(frame):
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected is a face (class_id for face)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    return boxes, confidences
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    # (Your existing code to merge frames and display them)

    boxes, confidences = detect_faces_yolo(frame)

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        label = str(ref.get(f"ID_{i:02}"))  # Assuming your database structure follows this format

        # Check if the detected face matches any enrolled student
        if label != "None":
            # Draw rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Update student's attendance status
            student_ref = db.reference(f"Students/ID_{i:02}")
            student_ref.update({"last_attendance_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "attendance_status": "Present"})

    # Display the frame with detected faces
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()









