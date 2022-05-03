import matplotlib.pyplot as plt
import torch
import h5py
import argparse
import numpy as np
from torch.utils.data import Dataset

class Ambisonic(Dataset):
    def __init__(self, path="sample/{}_train.pickle", transform=None, args=None, scene=None):
        self.data = h5py.File(path, "r")
        self.scenes = list(self.data.keys())
        if args.dry_run:
            self.scenes = self.scenes[:10]
        if scene is not None:
            self.scenes = [scene]
        self.transform = transform
        self.SC = 1024
        self.args = args
        self.eps = 1e-5

    def __len__(self):
        return len(self.scenes) * self.SC

    def __getitem__(self, idx):
        scene_idx = idx // self.SC
        sample_idx = idx % self.SC
        scene_name = self.scenes[scene_idx]
        scene_a = self.data[scene_name]['RAW']
        scene_b = self.data[scene_name]['TUNED']
        a_ran = self.data[scene_name]['raw_list'][sample_idx]
        b_ran = self.data[scene_name]['tuned_list'][sample_idx]

        #label = scene_b[:self.args.num_out_chan, b_ran[0]:b_ran[1]]
        label = scene_b[b_ran[0]:b_ran[1]]
        label = label[np.newaxis,...] / (np.max(np.abs(label)) + self.eps)

        image = scene_a[a_ran[0]:a_ran[1]]
        image = image[np.newaxis,...] / (np.max(np.abs(image)) + self.eps)
        return image, label


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
