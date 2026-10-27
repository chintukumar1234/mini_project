import os
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset


class BraTSDataset(Dataset):

    def __init__(self, data_dir, image_size=(96, 96, 96)):

        self.data_dir = data_dir
        self.image_size = image_size

        self.patients = []

        for patient in os.listdir(data_dir):

            patient_path = os.path.join(data_dir, patient)

            if os.path.isdir(patient_path):
                self.patients.append(patient_path)

        print("Total patients:", len(self.patients))

    def __len__(self):
        return len(self.patients)

    def load_nifti(self, path):

        image = nib.load(path)

        image = image.get_fdata()

        return image

    def normalize(self, image):

        image = image.astype(np.float32)

        mean = image.mean()
        std = image.std()

        image = (image - mean) / (std + 1e-8)

        return image

    def resize_3d(self, image):

        tensor = torch.tensor(image).unsqueeze(0).unsqueeze(0)

        tensor = torch.nn.functional.interpolate(
            tensor,
            size=self.image_size,
            mode="trilinear",
            align_corners=False
        )

        return tensor.squeeze().numpy()

    def __getitem__(self, index):

        patient_path = self.patients[index]

        files = os.listdir(patient_path)

        t1 = None
        t1ce = None
        t2 = None
        flair = None
        seg = None

        for file in files:

            file_path = os.path.join(patient_path, file)

            file_lower = file.lower()

            if "t1ce" in file_lower:
                t1ce = self.load_nifti(file_path)

            elif "flair" in file_lower:
                flair = self.load_nifti(file_path)

            elif "t2" in file_lower:
                t2 = self.load_nifti(file_path)

            elif "t1" in file_lower:
                t1 = self.load_nifti(file_path)

            elif "seg" in file_lower:
                seg = self.load_nifti(file_path)

        # Normalize MRI modalities
        t1 = self.normalize(t1)
        t1ce = self.normalize(t1ce)
        t2 = self.normalize(t2)
        flair = self.normalize(flair)

        # Resize MRI images
        t1 = self.resize_3d(t1)
        t1ce = self.resize_3d(t1ce)
        t2 = self.resize_3d(t2)
        flair = self.resize_3d(flair)

        # Resize segmentation mask
        seg_tensor = torch.tensor(seg).unsqueeze(0).unsqueeze(0)

        seg_tensor = torch.nn.functional.interpolate(
            seg_tensor.float(),
            size=self.image_size,
            mode="nearest"
        )

        seg = seg_tensor.squeeze().numpy()

        # Stack 4 MRI modalities
        image = np.stack(
            [t1, t1ce, t2, flair],
            axis=0
        )

        image = torch.tensor(image, dtype=torch.float32)

        seg = torch.tensor(seg, dtype=torch.long)

        return image, seg