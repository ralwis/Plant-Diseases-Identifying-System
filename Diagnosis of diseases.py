import cv2
import time
import numpy as np
import pyrebase

#Configuaration for Firebase
config = {
    'apiKey': "AIzaSyD63DeuxBAbrOnqv7glSivpaCPqf9XkaNs",
    'authDomain': "real-time-hydroponic.firebaseapp.com",
    'databaseURL': "https://real-time-hydroponic-default-rtdb.firebaseio.com",
    'projectId': "real-time-hydroponic",
    'storageBucket': "real-time-hydroponic.appspot.com",
    'messagingSenderId': "965004862660",
    'appId': "1:965004862660:web:7343907d4bfe7f65e79591",
    'measurementId': "G-XVXTY5LKBD"
}

firebase = pyrebase.initialize_app(config)
db = firebase.database()

def sendDataToFirebase(msg):
    print("Sending Data to the Firebase")

    data = {'Plant':msg}
    db.push(data)

    print("Records updated in the firebase")

def checkForDiseasedLeaves(image):
    result = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # boundary for Brown color range values
    lower1 = np.array([5, 125, 20])
    upper1 = np.array([15, 255, 255])

    # boundary for Yellow color range values
    lower2 = np.array([18, 125, 20])
    upper2 = np.array([32, 255, 255])

    lower_mask = cv2.inRange(image, lower1, upper1)
    upper_mask = cv2.inRange(image, lower2, upper2)

    full_mask = lower_mask + upper_mask;

    result = cv2.bitwise_and(result, result, mask=full_mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    for i in contours:
        cnt = cv2.contourArea(i)
        if cnt > 1000:
            cv2.drawContours(result, [i], 0, (0, 0, 0), -1)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        cv2.putText(result, 'Diseases Area =' + str(area), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1,cv2.LINE_AA)

        color_frame = cv2.resize(result, None, fx=0.8, fy=0.6)
        height, width, channels = color_frame.shape
        cv2.imshow('Color Identify Frame', color_frame)

        if area > 20:
            print("Diseases Found")
            return True
        else:
            print("Not enough evidence to identify Diseases")
            return False

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    print("\n-----------------------\nNew Image Captured")
    cam_frame = cv2.resize(frame, None, fx=0.8, fy=0.6)
    height, width, channels = cam_frame.shape
    cv2.imshow('Hydroponic Camera', cam_frame)

    # Load Yolo
    net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")

    classes = ["plant"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    img = cv2.resize(frame, None, fx=0.7, fy=0.5)
    height, width, channels = img.shape

    # Detecting objects
    print("Detecting Leaves...")
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                print("Plant identified with confidence of ", confidence)
                # Object detected

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

                if checkForDiseasedLeaves(frame):
                    sendDataToFirebase("Diseases Found")
                else:
                    sendDataToFirebase("Diseases Not Found")

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, label, (x, y + 30), font, 2, (0,255,0), 2)

    cv2.imshow("Plant Identify", img)
    cv2.waitKey(5000)