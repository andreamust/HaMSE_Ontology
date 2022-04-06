import msaf
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pypianoroll import load, Multitrack
import librosa

mp3 = "/Users/andrea/Documents/DHDK/Thesis/MIDI/Willie Peyote - Ogni giorno alle 18.mp3"
wav = "/Users/andrea/Documents/DHDK/Thesis/MIDI/Bach10_v1.1/02-AchLiebenChristen/02-AchLiebenChristen.wav"
# wav_1 = '/Users/andrea/Documents/DHDK/Thesis/MIDI/Bach_BWV849-01_001_20090916-SMD.wav'
wav_1 = '/Users/andrea/Documents/DHDK/Thesis/Ontology Data/audio.wav'

# mid = "/Users/andrea/Documents/DHDK/Thesis/MIDI/Bach10_v1.1/02-AchLiebenChristen/02-AchLiebenChristen.mid"
# mid_2 = Multitrack(mid)
# print(mid_2)

# print(msaf.features_registry)

MIN_SEGMENT_LEN = 30  #seconds

boundaries, labels = msaf.process(wav_1, boundaries_id="sf", labels_id="fmc2d", feature="pcp", plot=True)
new_boundaries = [boundaries[0]]
new_labels = [labels[0]]
for i in range(1, len(boundaries)):
    if (boundaries[i] - boundaries[i-1]) > MIN_SEGMENT_LEN:
        new_boundaries.append(boundaries[i])
        new_labels.append(labels[i])

print(labels)

sr = 44100
hop_length = 1024
beats_audio_file = wav
audio = librosa.load(beats_audio_file, sr=sr)[0]
audio_harmonic, audio_percussive = librosa.effects.hpss(audio)

# Compute beats
tempo, frames = librosa.beat.beat_track(y=audio_percussive,
                                        sr=sr, hop_length=hop_length)

# To times
beat_times = librosa.frames_to_time(frames, sr=sr,
                                    hop_length=hop_length)

print(tempo)
print(frames)
print(beat_times)
#print(new_labels, new_boundaries)
#print(msaf.get_all_boundary_algorithms())
#print(msaf.get_all_label_algorithms())
#print(msaf.features_registry)




