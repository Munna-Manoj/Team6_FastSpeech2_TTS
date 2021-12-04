import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
from utils import get_spectrograms, get_duration, get_energy, get_pitch
import hyperparams as hp
import librosa




class PrepareDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, csv_file, root_dir):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.

        """
        self.landmarks_frame = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir

    def load_wav(self, filename):
        return librosa.load(filename, sr=hp.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0]) + '.wav'
        mel, mag = get_spectrograms(wav_name)

        energy_raw = get_energy(wav_name)
        energy = (energy_raw-hp.energy_min)/(hp.energy_max - hp.energy_min)
        pitch_raw  = get_pitch(wav_name)
        pitch = (pitch_raw - hp.pitch_min)/(hp.pitch_max - hp.pitch_min)

        dur = get_duration(wav_name)
        #print(energy)
        #print(pitch)
        
        np.save(wav_name[:-4] + '.pt', mel)
        np.save(wav_name[:-4] + '.mag', mag)
        np.save(wav_name[:-4] + '.dur', dur)
        np.save(wav_name[:-4] + '.eng', energy)
        np.save(wav_name[:-4] + '.pth', pitch)

        sample = {'mel':mel, 'mag': mag, 'dur': dur, 'energy': energy, 'pitch':pitch}
        # + duration

        return sample
    
if __name__ == '__main__':
    dataset = PrepareDataset(os.path.join(hp.data_path,'metadata.csv'), os.path.join(hp.data_path,'wavs'))
    dataloader = DataLoader(dataset, batch_size=1, drop_last=False, num_workers=8)
    from tqdm import tqdm
    pbar = tqdm(dataloader)
    for d in pbar:
        pass
