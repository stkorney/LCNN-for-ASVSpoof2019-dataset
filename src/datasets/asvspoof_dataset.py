import os

import torch
import torchaudio
from torch.utils.data import Dataset

from src.datasets.base_dataset import BaseDataset


class ASVSpoofDataset(BaseDataset):
    """
    Dataset for ASVspoof 2019 Logical Access (LA) challenge.
    Loads audio files, converts them to spectrograms using STFT.
    """

    def __init__(
        self,
        data_dir,
        protocol_file,
        data_type="train",
        *args,
        **kwargs
    ):
        """
        Args:
            data_dir (str): path to directory containing audio files.
            protocol_file (str): path to protocol file with annotations.
            data_type (str): type of data.
        """
        self.data_dir = data_dir
        self.protocol_file = protocol_file
        self.data_type = data_type
        
        index = self._create_index()
        
        super().__init__(index, *args, **kwargs)

    def _create_index(self):
        """
        Create index from protocol file.
        
        Returns:
            index (list[dict]): list of dicts with 'path', 'label', and 'file_id' keys.
        """
        index = []
        with open(self.protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                file_name = parts[1] + '.flac'
                file_id = file_name.replace('.flac', '')
                # spoof = 0, bonafide = 1
                label = 0 if parts[4] == "spoof" else 1
                
                file_path = os.path.join(self.data_dir, file_name)
                if os.path.exists(file_path):
                    index.append({"path": file_path, "label": label, "file_id": file_id})
        
        return index

    def load_object(self, path):
        """
        Load audio file from dataset and convert to spectrogram.
        
        Args:
            path (str): path to audio file.
        Returns:
            spectrogram: spectrogram tensor.
        """
        inpt, sr = torchaudio.load(path)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            inpt = resampler(inpt)

        spectrogram = self._fft(inpt)
        
        return spectrogram

    def _fft(self, inpt):
        spec = torch.stft(input=inpt, n_fft=2048, hop_length=512, win_length=2048, window=torch.hann_window(2048), return_complex=True)
        spec = torch.log(torch.abs(spec) + 1e-9)

        _, freq, time = spec.shape
        if freq > 864:
            spec = spec[:, :864, :]
        else:
            spec = torch.nn.functional.pad(spec, (0, 0, 0, 864 - freq))
        if time > 600:
            spec = spec[:, :, :600]
        else:
            spec = torch.nn.functional.pad(spec, (0, 600 - time, 0, 0))
        
        return spec

