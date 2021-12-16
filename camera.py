import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


class Video(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.path = './IMAGE'

        self.images = []
        self.classNames = []

        myList = os.listdir(self.path)
        print(myList)
        for cl in myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        print(self.classNames)

        self.encodeListKnown = []
        for img in self.images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            self.encodeListKnown.append(encode)
        print('Encoding Complete')

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(
                self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(
                self.encodeListKnown, encodeFace)
            # print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = self.classNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2),
                              (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                # markAttendance(name)
                with open('Attendance.csv', 'r+') as f:
                    myDataList = f.readlines()
                    nameList = []
                    for line in myDataList:
                        entry = line.split(',')
                        nameList.append(entry[0])
                    if name not in nameList:
                        now = datetime.now()
                        tstr = now.strftime('%H:%M:%S')
                        dstr = now.strftime('%d/%m/%Y')
                        f.writelines(f'\n{name},{dstr},{tstr}')

        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
