from django.shortcuts import render
from django.http import StreamingHttpResponse
from keras.models import model_from_json
import cv2
import threading
import numpy as np

# Create your views here.
#@gip.gzip_page
def Home(request):
    #try:
    cam = VideoCamera()
    return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    #except:
        #pass
    return render(request, 'emotion_detection.html')

# video class
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame

        emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

        # load json and create model
        json_file = open('models/emotion_model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        emotion_model = model_from_json(loaded_model_json)

        # load weights into new model
        emotion_model.load_weights("models/emotion_model.h5")

        image = cv2.resize(image,(1280,720))
        face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(image, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

            # predict the emotions
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(image, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        _, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def update(self):
        while True:
            (self.grabbed, self.frame) = self.video.read()

def gen(camera):
    while True:
        output = camera.get_frame()
        yield (b' --frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n\r\n')