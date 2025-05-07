# import cv2
# import mediapipe as mp
# import time
# import numpy as np
# import os


# from numpy.ma.core import shape
# from sklearn.preprocessing import LabelEncoder
# from tensorflow.python.keras.utils.np_utils import to_categorical
# from keras.models import Sequential
# from keras.layers import TimeDistributed, Flatten, Input, LSTM, Dense, Dropout
# from sklearn.utils import shuffle
# from keras.callbacks import EarlyStopping

# cap = cv2.VideoCapture(0)

# mpHands = mp.solutions.hands
# hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# prevTime = 0
# currTime = 0
# capture_coordinates = False
# i = 0
# vector = []
# vector2 = []

# directory = os.path.dirname(__file__)

# while 0:
#     success, img = cap.read()
#     imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     imgRGB = np.array(imgRGB, dtype=np.uint8)   # Zmiana typu danych, żeby mediapipe działało
#     results = hands.process(imgRGB)
#     if results.multi_hand_landmarks:
#         for handLms in results.multi_hand_landmarks:
#             if capture_coordinates:
#                 print("i = ", i)
#                 i += 1
#                 vector = []
#                 for id, lm in enumerate(handLms.landmark):  # Zapisywanie współrzędnych do wektora
#                     height,width,channels = img.shape
#                     print(f"Punkt {id}: ({lm.x}, {lm.y}, {lm.z})")
#                     vector.append([lm.x,lm.y,lm.z]) 
#                 vector2.append(vector)
#             mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)   # Wyświetlanie punktów i połączeń między nimi

#     currTime = time.time()  
#     fps = 1/(currTime-prevTime) # Obliczanie FPS
#     prevTime = currTime

#     cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
#     cv2.imshow("Image", img)
#     if i >= 60:     # Zapisywanie 60 klatek do folderu
#         i = 0
#         capture_coordinates = False

#         np.save(directory+"/dane/"+letter, vector2)
#         print(vector2)

#         vector2 = []


#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('r'):     # Nadawanie nazwy literze
#         letter = input("Jaką literę chcesz zapisać? ")
#         time.sleep(1)
#         capture_coordinates = True
#     elif key == ord('q'):   # Wychodzenie z programu
#         break

# cap.release()
# cv2.destroyAllWindows()



# rozpoznawanie_liter = os.path.dirname(directory)
# pslr = os.path.dirname(rozpoznawanie_liter)

# #### KONWERSJA NA 60 KLATEK #################

# # data_path = os.path.join(pslr, 'data')
# # for letter in os.listdir(data_path):
# #     curr_path = os.path.join(data_path, letter)
# #     for num in os.listdir(curr_path):
# #         curr_patha = os.path.join(curr_path, num)
# #         data = np.load(curr_patha, allow_pickle=True)
# #
# #         frames_over_0_30_60 = shape(data)[0]%30
# #         frames = shape(data)[0]
# #
# #         if 30 <= frames < 60:
# #             data_duplicated = data[:-(60 - frames)]
# #             for i in range(60 - frames, 0, -1):
# #                 last_frame = data[-i]
# #                 duplicates = np.repeat(last_frame[np.newaxis, :, :], 2, axis=0)
# #                 data_duplicated = np.concatenate((data_duplicated, duplicates), axis=0)
# #             data = data_duplicated
# #
# #         elif frames > 60:
# #             data = data[:-frames_over_0_30_60]
# #
# #         # print(curr_patha, shape(data))
# #         print(letter + '/' + num)
# #
# #         save_folder_path = os.path.join(pslr, 'data60')
# #         save_letter_folder_path = os.path.join(save_folder_path, letter)
# #         save_file_path = os.path.join(save_letter_folder_path, num)
# #
# #         os.makedirs(save_letter_folder_path, exist_ok=True)
# #         np.save(save_file_path, data)

# #####################################################################



# #### WYŚWIETLANIE PRZYKŁADOWYCH DŁONI #################

# import matplotlib.pyplot as plt
# # from mpl_toolkits.mplot3d import Axes3D

# # # Wczytanie jednej przykładowej próbki z zestawu
# # X = []
# # y = []
# # data_path = os.path.join(pslr, 'data60')
# # for letter in os.listdir(data_path):
# #     curr_path = os.path.join(data_path, letter)
# #     for num in os.listdir(curr_path):
# #         curr_patha = os.path.join(curr_path, num)
# #         data = np.load(curr_patha, allow_pickle=True)

# #         X.append(data)
# #         y.append(letter)

# # samples = []  # shape: (60, 21, 3)
# # for indeks in np.arange(0, 1):
# #     samples.append(X[indeks])

# # # Połączenia punktów dłoni
# # connections = [
# #     (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
# #     (0, 5), (5, 6), (6, 7), (7, 8),      # Index
# #     (0, 9), (9, 10), (10, 11), (11, 12), # Middle
# #     (0, 13), (13, 14), (14, 15), (15, 16), # Ring
# #     (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
# # ]


# # # Dla każdej klatki: narysuj punkty i połączenia
# # for sample in samples:
# #     fig = plt.figure(figsize=(10, 8))
# #     ax = fig.add_subplot(111, projection='3d')
# #     ax.set_title("Wszystkie 60 klatek jednej próbki")
# #     ax.set_xlabel('X')
# #     ax.set_ylabel('Y')
# #     ax.set_zlabel('Z')

# #     for frame in sample:
# #         x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
# #         ax.scatter(x, y, z, c='blue', s=5)
# #         for start, end in connections:
# #             xs, ys, zs = frame[[start, end]].T
# #             ax.plot(xs, ys, zs, c='gray', linewidth=0.5)

# # plt.tight_layout()
# # plt.show()




















# X = []
# y = []

# data_path = os.path.join(pslr, 'data60')
# for letter in os.listdir(data_path):
#     curr_path = os.path.join(data_path, letter)
#     for num in os.listdir(curr_path):
#         curr_patha = os.path.join(curr_path, num)
#         data = np.load(curr_patha, allow_pickle=True)

#         X.append(data)
#         y.append(letter)

# X = np.array(X)
# y = np.array(y)
# print(X.shape)

# ls = LabelEncoder()
# y_encoded = ls.fit_transform(y)
# y_onehot = to_categorical(y_encoded, num_classes=36)
# X, y_onehot = shuffle(X, y_onehot, random_state=42)


# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)


# model = Sequential()
# model.add(Input((60, 21, 3)))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(256, return_sequences=True))
# model.add(LSTM(128))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(36, activation='softmax'))

# model.compile(optimizer='adamw', loss='categorical_crossentropy', metrics=['accuracy'])

# es = EarlyStopping(patience=7, restore_best_weights=True)
# #history = model.fit(X, y_onehot, epochs=10, batch_size=64, validation_split=0.2, callbacks=[es])
# history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.2, callbacks=[es])







# # Accuracy
# plt.figure("Accuracy")
# plt.plot(history.history['accuracy'], label='Train accuracy')
# plt.plot(history.history['val_accuracy'], label='Val accuracy')
# plt.title('Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()

# # Loss
# plt.figure("Loss")
# plt.plot(history.history['loss'], label='Train loss')
# plt.plot(history.history['val_loss'], label='Val loss')
# plt.title('Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()

# plt.tight_layout()
# plt.show()

# # Predykcja na zbiorze testowym
# y_pred = model.predict(X_test)
# y_pred_labels = np.argmax(y_pred, axis=1)
# y_true_labels = np.argmax(y_test, axis=1)

# # Wyświetlenie przykładowych predykcji
# for i in range(20):
#     print(f"Expected: {ls.inverse_transform([y_true_labels[i]])[0]}, Predicted: {ls.inverse_transform([y_pred_labels[i]])[0]}")




























import cv2
import mediapipe as mp
import time
import numpy as np
import os
import string


from numpy.ma.core import shape
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import TimeDistributed, Flatten, Input, LSTM, Dense, Dropout
from sklearn.utils import shuffle
from keras.callbacks import EarlyStopping

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

directory = os.path.dirname(__file__)


while 0:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgRGB = np.array(imgRGB, dtype=np.uint8)   # Zmiana typu danych, żeby mediapipe działało
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            if capture_coordinates:
                print("i = ", i)
                i += 1
                vector = []
                for id, lm in enumerate(handLms.landmark):  # Zapisywanie współrzędnych do wektora
                    height,width,channels = img.shape
                    print(f"Punkt {id}: ({lm.x}, {lm.y}, {lm.z})")
                    vector.append([lm.x,lm.y,lm.z]) 
                vector2.append(vector)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)   # Wyświetlanie punktów i połączeń między nimi

    currTime = time.time()  
    fps = 1/(currTime-prevTime) # Obliczanie FPS
    prevTime = currTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    if i >= 60:     # Zapisywanie 60 klatek do folderu
        i = 0
        capture_coordinates = False

        np.save(directory+"/dane/"+letter, vector2)
        print(vector2)

        vector2 = []


    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):     # Nadawanie nazwy literze
        letter = input("Jaką literę chcesz zapisać? ")
        time.sleep(1)
        capture_coordinates = True
    elif key == ord('q'):   # Wychodzenie z programu
        break

cap.release()
cv2.destroyAllWindows()



rozpoznawanie_liter = os.path.dirname(directory)
pslr = os.path.dirname(rozpoznawanie_liter)

#### KONWERSJA NA 60 KLATEK #################

# data_path = os.path.join(pslr, 'data')
# for letter in os.listdir(data_path):
#     curr_path = os.path.join(data_path, letter)
#     for num in os.listdir(curr_path):
#         curr_patha = os.path.join(curr_path, num)
#         data = np.load(curr_patha, allow_pickle=True)
#
#         frames_over_0_30_60 = shape(data)[0]%30
#         frames = shape(data)[0]
#
#         if 30 <= frames < 60:
#             data_duplicated = data[:-(60 - frames)]
#             for i in range(60 - frames, 0, -1):
#                 last_frame = data[-i]
#                 duplicates = np.repeat(last_frame[np.newaxis, :, :], 2, axis=0)
#                 data_duplicated = np.concatenate((data_duplicated, duplicates), axis=0)
#             data = data_duplicated
#
#         elif frames > 60:
#             data = data[:-frames_over_0_30_60]
#
#         # print(curr_patha, shape(data))
#         print(letter + '/' + num)
#
#         save_folder_path = os.path.join(pslr, 'data60')
#         save_letter_folder_path = os.path.join(save_folder_path, letter)
#         save_file_path = os.path.join(save_letter_folder_path, num)
#
#         os.makedirs(save_letter_folder_path, exist_ok=True)
#         np.save(save_file_path, data)

#####################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Wczytanie jednej przykładowej próbki z zestawu
X = []
y = []
data_path = os.path.join(pslr, 'data60')
for letter in os.listdir(data_path):
    curr_path = os.path.join(data_path, letter)
    for num in os.listdir(curr_path):
        curr_patha = os.path.join(curr_path, num)
        data = np.load(curr_patha, allow_pickle=True)

        X.append(data)
        y.append(letter)

samples = []  # shape: (60, 21, 3)
for indeks in np.arange(0, 1):
    samples.append(X[indeks])

# Połączenia punktów dłoni
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]


# Dla każdej klatki: narysuj punkty i połączenia
for sample in samples:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title("Wszystkie 60 klatek jednej próbki")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    for frame in sample:
        x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
        ax.scatter(x, y, z, c='blue', s=5)
        for start, end in connections:
            xs, ys, zs = frame[[start, end]].T
            ax.plot(xs, ys, zs, c='gray', linewidth=0.5)

plt.tight_layout()
plt.show()




# X = []
# y = []

# data_path = os.path.join(pslr, 'data60')
# for letter in os.listdir(data_path):
#     curr_path = os.path.join(data_path, letter)
#     for num in os.listdir(curr_path):
#         curr_patha = os.path.join(curr_path, num)
#         data = np.load(curr_patha, allow_pickle=True)

#         X.append(data)
#         y.append(letter)

# X = np.array(X)
# y = np.array(y)
# print(X.shape)

# ls = LabelEncoder()
# y_encoded = ls.fit_transform(y)
# y_onehot = to_categorical(y_encoded, num_classes=36)
# X, y_onehot = shuffle(X, y_onehot, random_state=42)

# model = Sequential()
# model.add(Input((60, 21, 3)))
# model.add(TimeDistributed(Flatten()))
# model.add(LSTM(256, return_sequences=True))
# model.add(LSTM(128))
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(36, activation='softmax'))

# model.compile(optimizer='adamw', loss='categorical_crossentropy', metrics=['accuracy'])

# es = EarlyStopping(patience=7, restore_best_weights=True)
# model.fit(X, y_onehot, epochs=50, batch_size=64, validation_split=0.2, callbacks=[es])



X = []
y = []

data_path = os.path.join(pslr, 'data60')
for letter in os.listdir(data_path):
    curr_path = os.path.join(data_path, letter)
    for num in os.listdir(curr_path):
        curr_patha = os.path.join(curr_path, num)
        data = np.load(curr_patha, allow_pickle=True)

        X.append(data)
        y.append(letter)

X = np.array(X)
y = np.array(y)
print(X.shape)

ls = LabelEncoder()
y_encoded = ls.fit_transform(y)
y_onehot = to_categorical(y_encoded, num_classes=36)
X, y_onehot = shuffle(X, y_onehot, random_state=42)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)


model = Sequential()
model.add(Input((60, 21, 3)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(36, activation='softmax'))

model.compile(optimizer='adamw', loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(patience=7, restore_best_weights=True)
#history = model.fit(X, y_onehot, epochs=5, batch_size=64, validation_split=0.2, callbacks=[es])
history = model.fit(X_train, y_train, epochs=50, batch_size=64, validation_split=0.25, callbacks=[es])

# Accuracy
plt.figure("Accuracy")
plt.plot(history.history['accuracy'], label='Train accuracy')
plt.plot(history.history['val_accuracy'], label='Val accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.figure("Loss")
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Predykcja na zbiorze testowym
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

# Wyświetlenie przykładowych predykcji
for i in range(len(y_pred)):
    print(f"Expected: {ls.inverse_transform([y_true_labels[i]])[0]}, Predicted: {ls.inverse_transform([y_pred_labels[i]])[0]}")


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true_labels, y_pred_labels)
print(f"\nDokładność predykcji: {accuracy * 100:.2f}%")