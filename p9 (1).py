import cv2
import numpy as np
from keras.models import load_model
m1 = load_model("C:/Users/91797/OneDrive/Desktop/m1.h5")
cap = cv2.VideoCapture(0)
array = []
hm = {}
t = 97
for i in range(0, 26):
    hm[i] = chr(t)
    t += 1
while True:
    success, imgOrignal = cap.read()
    img = np.array(imgOrignal)
    img = cv2.resize(img, (80, 80))
    img = np.expand_dims(img, axis=0)
    print(hm[np.argmax(m1.predict(img))])

