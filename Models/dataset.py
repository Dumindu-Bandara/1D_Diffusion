import torch
import torchaudio
from torchaudio import transforms as T
import random
from glob import glob
import os
from utils import Stereo, PadCrop, RandomPhaseInvert
import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial
import numpy as np

class SampleDataset(torch.utils.data.Dataset):
  def __init__(self, data_path):
    super().__init__()
    data = np.load(data_path)
    self.data = torch.from_numpy(data)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    out = self.data[idx].repeat(2, 1)
    out = out.type(torch.HalfTensor)
    return (out, idx)