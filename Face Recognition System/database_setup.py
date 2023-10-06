import os
import pickle
import numpy as np
import cv2
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials,db, storage
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials


cred = credentials.Certificate("face recog.json")
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://faceattendance-7e387-default-rtdb.firebaseio.com/',
    'storageBucket':'faceattendance-7e387.appspot.com'
})
bucket = storage.bucket()
ref = db.reference("Students")

data = {
    "ID_00_Felix":
        {
            "name": "Felix Pewipie",
            "major": "Gaming",
            "starting_year": 2017,
            "total_attendance": 7,
            "standing": "G",
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "ID_01_Emily":
        {
            "name": "Emly Blunt",
            "major": "Economics",
            "starting_year": 2021,
            "total_attendance": 12,
            "standing": "B",
            "year": 1,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "ID_02_Elon":
        {
            "name": "Elon Musk",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "standing": "G",
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}
img_path= "Images"
file_list= os.listdir(img_path)
img_list = []
student_id = []

for file in file_list:
    img = cv2.imread(os.path.join(img_path,file))
    img_list.append(img)
    #print(file)

    #remove the extention from img file name
    #print(os.path.splitext(file)[0])
    student_id.append(os.path.splitext(file)[0])

    FileName = f'{img_path}/{file}'
    bucket = storage.bucket()
    blob = bucket.blob(FileName)
    blob.upload_from_filename(FileName)
for key, val in data.items():
    ref.child(key).set(val)

