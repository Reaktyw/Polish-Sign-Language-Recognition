import numpy as np
import os
import matplotlib.pyplot as plt


# Połączenia punktów dłoni
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

directory = os.path.dirname(__file__)
rozpoznawanie_liter = os.path.dirname(directory)
pslr = os.path.dirname(rozpoznawanie_liter)

def data60_display(character, amount):
    datas = []
    plot_names = []
    data_path = os.path.join(pslr, 'data60', character)


    for num in os.listdir(data_path):
        curr_path = os.path.join(data_path, num)
        data = np.load(curr_path, allow_pickle=True)

        datas.append(data[40:60])
        plot_names.append(num)

    samples = []  # shape: (60, 21, 3)
    for indeks in np.arange(-amount, 0):
        samples.append(datas[indeks])


    # Dla każdej klatki: narysuj punkty i połączenia
    i = 1
    for sample in samples:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Wszystkie 60 klatek jednej próbki '{plot_names[-i]}'")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for frame in sample:
            x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
            ax.scatter(x, y, z, c='blue', s=5)
            for start, end in connections:
                xs, ys, zs = frame[[start, end]].T
                ax.plot(xs, ys, zs, c='gray', linewidth=0.5)
        i = i + 1

    plt.tight_layout()
    plt.show()



def dane_display(character, amount):
    datas = []
    plot_names = []
    data_path = os.path.join(directory, 'dane', character)
    for num in os.listdir(data_path):
        curr_path = os.path.join(data_path, num)
        data = np.load(curr_path, allow_pickle=True)

        datas.append(data)
        plot_names.append(num)


    samples = []  # shape: (60, 21, 3)
    for indeks in np.arange(-amount, 0):
        samples.append(datas[indeks])


    # Dla każdej klatki: narysuj punkty i połączenia
    i = 1
    for sample in samples:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f"Wszystkie 60 klatek jednej próbki '{plot_names[-i]}'")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        for frame in sample:
            x, y, z = frame[:, 0], frame[:, 1], frame[:, 2]
            ax.scatter(x, y, z, c='blue', s=5)
            for start, end in connections:
                xs, ys, zs = frame[[start, end]].T
                ax.plot(xs, ys, zs, c='gray', linewidth=0.5)
        i = i + 1

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    data60_display('ź', 5)
    dane_display('ź', 1)