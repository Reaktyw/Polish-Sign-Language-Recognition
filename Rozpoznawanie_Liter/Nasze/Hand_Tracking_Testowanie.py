from random import random, randrange
import cv2
import mediapipe as mp
import time

import numpy as np
import os
from sklearn.preprocessing import LabelEncoder

from Rozpoznawanie_Liter.Nasze.HandsDetection import HandsDetection
from Rozpoznawanie_Liter.Nasze.LetterPreparation import LetterPreparation
from Rozpoznawanie_Liter.Nasze.PSLR_Model import PSLR_Model

directory = os.path.dirname(__file__)
ls = LabelEncoder()



# letter_preparation = LetterPreparation(directory)
# letter_preparation.convert_data_to_60_frames()

# letter_preparation.prepare_data_for_training(ls)
PSLR_model = PSLR_Model(ls)
# PSLR_model.train_model(letter_preparation.X, letter_preparation.Y,f'{directory}/model_test_04_06_2025_1.keras')



modell = PSLR_model.load_model(f'{directory}/model_fine_tuning_04_06_2025_1.keras')
hd = HandsDetection()
hd.capture(recognition=True, model= modell, dir= directory)