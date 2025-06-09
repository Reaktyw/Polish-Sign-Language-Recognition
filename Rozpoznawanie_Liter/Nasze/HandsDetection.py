import pickle
import cv2
import mediapipe as mp
import time

import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image



class HandsDetection:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()
        self.Draw = mp.solutions.drawing_utils
        self.capture_coordinates = True
        self.directory = os.path.dirname(__file__)

    def capture(self, recognition= False, model= None, dir= None):
        with open(f'{dir}/label_encoder_1.pkl', 'rb') as f:
            ls = pickle.load(f)
        print(recognition)
        prevTime = 0
        i=0
        X = []
        fps = 30
        frame_duration = 1/fps
        last_predicted = ''
        while 1:
            start_time = time.perf_counter()
            success, img = self.cap.read()
            img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_RGB = np.array(img_RGB, dtype=np.uint8)
            results = self.hands.process(img_RGB)
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    if self.capture_coordinates:
                        vector = []
                        for id, lm in enumerate(handLms.landmark):  # Zapisywanie współrzędnych do wektora
                            height,width,channels = img.shape
                            vector.append([lm.x,lm.y,lm.z])
                        X.append(vector)

                        if recognition:
                            if len(X) > 60:
                                X = X[20:]

                            if len(X) == 60 and i % 10 == 0:
                                X_array = np.array(X)  # (60, 21, 3)
                                X_array = np.expand_dims(X_array, axis=0)
                                print(X_array.shape)

                                # Predictowanie
                                y_pred = model.predict(X_array)
                                y_pred_labels = np.argmax(y_pred, axis=1)
                                predicted_index = y_pred_labels[0]
                                predicted = ls.inverse_transform([predicted_index])[0]
                                confidence = y_pred[0][predicted_index] * 100
                                print(f"Predicted: {predicted}, Confidence: {confidence:.2f}%")

                                if confidence >= 95:
                                    last_predicted = predicted




                    self.Draw.draw_landmarks(img, handLms, self.mp_hands.HAND_CONNECTIONS)   # Wyświetlanie punktów i połączeń między nimi

            currTime = time.time()
            real_fps = 1/(currTime-prevTime)
            prevTime = currTime

            if recognition:
                img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(img_pil)
                font = ImageFont.truetype("arial.ttf", 40)  # Upewnij się, że font istnieje
                draw.text((80, 80), f"{last_predicted}", font=font, fill=(255, 0, 255))  # kolor RGB
                img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cv2.putText(img, str(int(real_fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
            cv2.imshow("Image", img)



            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):   # Wychodzenie z programu
                break
            elapsed_time = time.perf_counter() - start_time
            wait_time = max(0, frame_duration - elapsed_time)
            time.sleep(wait_time)
        self.cap.release()
        cv2.destroyAllWindows()