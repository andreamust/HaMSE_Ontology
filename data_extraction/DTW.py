from __future__ import print_function
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# mfrom midi2audio import FluidSynth
# import librosa
import librosa.display

mid = "/Users/andrea/Documents/DHDK/Thesis/MIDI/Bach10_v1.1/02-AchLiebenChristen/02-AchLiebenChristen.mid"
out = "/Users/andrea/Documents/DHDK/Thesis/MIDI/midi.wav"
# using the default sound font in 44100 Hz sample rate
#fs = FluidSynth()
#fs.midi_to_audio(mid, out)

wav_1 = '/Users/andrea/Documents/DHDK/Thesis/Ontology Data/audio.wav'
wav_2 = '/Users/andrea/Documents/DHDK/Thesis/Ontology Data/exported.wav'

x_1, fs = librosa.load(wav_1)
plt.figure(figsize=(16, 4))
librosa.display.waveplot(x_1, sr=fs)
plt.title('original')
plt.tight_layout()

x_2, fs = librosa.load(wav_2)
plt.figure(figsize=(16, 4))
librosa.display.waveplot(x_2, sr=fs)
plt.title('from_MIDI')
plt.tight_layout()


n_fft = 4410
hop_size = 2205

x_1_chroma = librosa.feature.chroma_stft(y=x_1, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)
x_2_chroma = librosa.feature.chroma_stft(y=x_2, sr=fs, tuning=0, norm=2,
                                         hop_length=hop_size, n_fft=n_fft)

plt.figure(figsize=(16, 8))
plt.subplot(2, 1, 1)
plt.title('Chroma Representation of $X_1$')
librosa.display.specshow(x_1_chroma, x_axis='time',
                         y_axis='chroma', cmap='gray_r', hop_length=hop_size)
plt.colorbar()
plt.subplot(2, 1, 2)
plt.title('Chroma Representation of $X_2$')
librosa.display.specshow(x_2_chroma, x_axis='time',
                         y_axis='chroma', cmap='gray_r', hop_length=hop_size)
plt.colorbar()
plt.tight_layout()

D, wp = librosa.sequence.dtw(X=x_1_chroma, Y=x_2_chroma, subseq=True)
wp_s = np.asarray(wp) * hop_size / fs

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
librosa.display.specshow(D, x_axis='time', y_axis='time',
                         cmap='gray_r', hop_length=hop_size)
imax = ax.imshow(D, cmap=plt.get_cmap('gray_r'),
                 origin='lower', interpolation='nearest', aspect='auto')
ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
plt.title('Warping Path on Acc. Cost Matrix $D$')
plt.colorbar()

fig = plt.figure(figsize=(16, 8))

"""----------------------------------------------------------"""

# Plot x_1
plt.subplot(2, 1, 1)
librosa.display.waveplot(x_1, sr=fs)
plt.title('From MIDI')
ax1 = plt.gca()

# Plot x_2
plt.subplot(2, 1, 2)
librosa.display.waveplot(x_2, sr=fs)
plt.title('Original')
ax2 = plt.gca()

plt.tight_layout()

trans_figure = fig.transFigure.inverted()
lines = []
arrows = 500
points_idx = np.int16(np.round(np.linspace(0, wp.shape[0] - 1, arrows)))
# print(points_idx)
# print(wp)
# print(wp[points_idx])
# for tp1, tp2 in zip((wp[points_idx, 0]) * hop_size, (wp[points_idx, 1]) * hop_size):
for tp1, tp2 in wp[points_idx] * hop_size / fs:
    print(tp1, tp2)
    # get position on axis for a given index-pair
    coord1 = trans_figure.transform(ax1.transData.transform([tp1, 0]))
    coord2 = trans_figure.transform(ax2.transData.transform([tp2, 0]))

    # draw a line
    line = matplotlib.lines.Line2D((coord1[0], coord2[0]),
                                   (coord1[1], coord2[1]),
                                   transform=fig.transFigure,
                                   color='r')
    lines.append(line)

fig.lines = lines
plt.tight_layout()

plt.show()
