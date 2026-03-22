import pathlib
import cv2
import torch
from torch.utils.data import Dataset

class UCSDDataset(Dataset):
    def __init__(self,
                 root_dir,
                 sequence_length=4,
                 resize=(128, 128)):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.resize = resize

        self.sequences = self._load_sequences()

    def _load_sequences(self):
        sequences = []
        try:
            folders = sorted([x for x in pathlib.Path(self.root_dir).iterdir() if x.is_dir()])
        except Exception as e:
            print(f"Невозможно загрузить папки, ошибка {str(e)}")
            return
        for folder in folders:
            frames = sorted(folder.iterdir())

            for i in range(len(frames) - self.sequence_length):
                sequences.append(frames[i:i+self.sequence_length])
        return sequences

    def __len__(self):
        return len(self.sequences)

    def _read_frame(self, path):
        try:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, self.resize)
        except Exception as e:
            print(f"Проблема с загрузкой изображения файл ucsd.py, ошибка {str(e)}")
            return
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img)
        img = img.unsqueeze(0)
        return img

    def __getitem__(self, item):
        paths = self.sequences[item]
        frames = []
        for p in paths:
            frame = self._read_frame(p)
            frames.append(frame)
        frames = torch.stack(frames)
        return frames