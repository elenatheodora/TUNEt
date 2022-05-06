# Elena Georgieva
# Vocal Tunign Project: Detune Audio
# DS 1008
# Spring 2022


import numpy as np
import librosa
import pandas as pd
import IPython.display as ipd
import matplotlib.pyplot as plt
import librosa.display
import os
import subprocess
from IPython.display import Audio
import base64
import pedalboard
from scipy.io.wavfile import write


## Load audio files
fs = 44100  
path = "/Users/elenageorgieva/Desktop/vtd/train/"    
raw_path = path + "raw/" # raw input data, not touching tuned output or val set
shifted_path = path + "shifted/" # path to save shifted data. 

raw_filenames = os.listdir(raw_path)
all_filenames = [f for f in raw_filenames[19:len(raw_filenames)]] # raw_filenames[0:3] to test first few

# Apply Pedalboard Effect
pitch = pedalboard.PitchShift()
semitones_arr = [0.1, 0.2, 0.3, 0.4, 0.5, -0.1, -0.2, -0.3, -0.4, -0.5]

progress_bar = 19

for file in all_filenames:
    progress_bar += 1
    print("---Progress: " + str(progress_bar) + "---")
    if (file == ".DS_Store"): # weird error, ignore .DS_Store 
        continue
    for amount in semitones_arr:
        y, fs = librosa.load(raw_path + file)
        # write(shifted_path + file, fs, y) # to write wav file shifted 0, not necessary
        pitch.semitones = amount
        shifted = pitch(y,sample_rate = fs)
        Audio(shifted, rate=fs)
        write(shifted_path + file + "_" + str(amount) + ".wav", fs, shifted)

