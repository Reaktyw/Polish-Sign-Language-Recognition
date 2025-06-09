import os
from random import random, randrange
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from numpy import shape
from sklearn.utils import shuffle


class LetterPreparation:
    def __init__(self, dir):
        self.rozpoznawanie_liter = os.path.dirname(dir)
        self.pslr = os.path.dirname(self.rozpoznawanie_liter)
        self.data_path = os.path.join(self.pslr, 'data')
        self.X = None
        self.Y = None

    def convert_data_to_60_frames(self):
        for letter in os.listdir(self.data_path):
            curr_path = os.path.join(self.data_path, letter)
            for num in os.listdir(curr_path):
                curr_patha = os.path.join(curr_path, num)
                data = np.load(curr_patha, allow_pickle=True)

                frames_over_0_30_60 = shape(data)[0]%30
                frames = shape(data)[0]

                if 30 <= frames < 60:
                    data_duplicated = data[:-(60 - frames)]
                    for i in range(60 - frames, 0, -1):
                        last_frame = data[-i]
                        duplicates = np.repeat(last_frame[np.newaxis, :, :], 2, axis=0)
                        data_duplicated = np.concatenate((data_duplicated, duplicates), axis=0)
                    data = data_duplicated

                elif frames > 60:
                    data = data[:-frames_over_0_30_60]

                save_folder_path = os.path.join(self.pslr, 'data60')
                save_letter_folder_path = os.path.join(save_folder_path, letter)
                save_file_path = os.path.join(save_letter_folder_path, num)

                os.makedirs(save_letter_folder_path, exist_ok=True)
                np.save(save_file_path, data)

    def prepare_data_for_training(self, ls):
        X = []
        Y = []
        data_path = os.path.join(self.pslr, 'data60')

        for letter in os.listdir(data_path):
            curr_path = os.path.join(data_path, letter)
            for num in os.listdir(curr_path):
                curr_patha = os.path.join(curr_path, num)
                data = np.load(curr_patha, allow_pickle=True)

                X.append(data)
                Y.append(letter)
                ran = (randrange(0, 2) -1) / 100
                data = data[:, :, :] + ran

                scale = (randrange(5, 15)) /100 + 1
                data = data[:, :, :] * scale

                angle = np.radians((randrange(100, 150)) / 10)
                for i in range(0,60):
                    wrist_position = data[i, 0, :2]
                    wrist_centre = data[i, :, :2] - wrist_position
                    x_rotated = wrist_centre[:, 0] * np.cos(angle) - wrist_centre[:, 1] * np.sin(angle)
                    y_rotated = wrist_centre[:, 0] * np.sin(angle) + wrist_centre[:, 1] * np.cos(angle)
                    data[i, :, 0] = x_rotated + wrist_position[0]
                    data[i, :, 1] = y_rotated + wrist_position[1]
                X.append(data)
                Y.append(letter)
        X = np.array(X)
        Y = np.array(Y)

        y_encoded = ls.fit_transform(Y)
        y_onehot = to_categorical(y_encoded, num_classes=36)
        self.X, self.Y = shuffle(X, y_onehot, random_state=42)