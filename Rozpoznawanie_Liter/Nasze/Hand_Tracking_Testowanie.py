import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
prevTime = 0
currTime = 0
capture_coordinates = False
i = 0
vector = []
vector2 = []
folder = "Rozpoznawanie_Liter/Nasze/dane/"

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if capture_coordinates:
                print("i = ", i)
                i += 1
                vector = []
                for id, lm in enumerate(handLms.landmark):
                    height,width,channels = img.shape
                    #cx, cy = int(lm.x*width), int(lm.y*height) TEGO UŻYWAĆ JEŻELI CHCEMY TYLKO X I Y (ALE NIE CHCEMY)
                    print(f"Punkt {id}: ({lm.x}, {lm.y}, {lm.z})")
                    vector.append([lm.x,lm.y,lm.z])
                vector2.append(vector)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    currTime = time.time()
    fps = 1/(currTime-prevTime)
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    if i >= 60:
        i = 0
        capture_coordinates = False

        np.save(folder+letter, vector2)
        print(vector2)

        vector2 = []


    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        letter = input("Jaką literę chcesz zapisać? ")
        time.sleep(1)
        capture_coordinates = True
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
