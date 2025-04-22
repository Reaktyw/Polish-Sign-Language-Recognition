import numpy as np
folder = "Rozpoznawanie_Liter/Nasze/dane/a.npy"
a1 = np.load(folder)

folder = "data/ą/0.npy"
a2 = np.load(folder)

print(f"Porównanie naszego i ichniego: \n {a1}, \n \n {a2}")
print("cos tam")
