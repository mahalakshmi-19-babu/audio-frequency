#my audio comparison scrit for project
#i wrote this to check two voices

import os
import numpy as np #for numbers
from scipy.io.wavfile import read as wav_read #to read wav
from scipy.signal import spectrogram #for freq graph
from scipy.spatial.distance import cosine #for similarity
import matplotlib.pyplot as plt #for plotting

print("my voice test begin")


#my audio files
voice_file_one =r"C:/Users/DELL/Desktop/audio.frequency.py/voice1.wav.wav"
voice_file_two =r"C:/Users/DELL/Desktop/audio.frequency.py/voice2.wav.wav"

if not os.path.exists(voice_file_one):
    print("hey,first file not here")
    quit()
    
if not os.path.exists(voice_file_two):
    print("second file is missing")
    quit()

sample_rate1,audio_data1 = wav_read(voice_file_one)
sample_rate2,audio_data2 = wav_read(voice_file_two)

#make same size
shortest = min(len(audio_data1), len(audio_data2))
audio_data1 = audio_data1[:shortest]
audio_data2 = audio_data2[:shortest]

#if stereo make mono
if len(audio_data1.shape)==2:
    audio_data1 = np.mean(audio_data1,axis=1)
if len(audio_data2.shape)==2:
    audio_data2 = np.mean(audio_data2,axis=1)

#get spectro
freq1, times1, spec1 = spectrogram(audio_data1, sample_rate1, nperseg=1024)
freq2, times2, spec2 = spectrogram(audio_data2, sample_rate2, nperseg=2048)

#plot now
fig = plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.pcolormesh(times1, freq1, np.log10(spec1 + 1e-10)*10,cmap='viridis',shading='nearest')
plt.title('first voice freq')
plt.xlabel('time in sec')
plt.ylabel('freq hz')
plt.colorbar().set_label('power dp')

plt.subplot(122)
plt.pcolormesh(times2, freq2, np.log10(spec2 + 1e-10)*10,cmap='viridis',shading='nearest')
plt.title('second voice freq')
plt.xlabel('time in sec')
plt.ylabel('freq hz')
plt.colorbar().set_label('dp')

plt.suptitle('my voice compares')
plt.tight_layout(pad=1.5)
plt.savefig('my_graph.png')
plt.show()

print("graph done,check my_graph.png")

logspec1 = np.log10(spec1+1e-10).ravel()
log_spec_2 = np.log10(spec2+1e-9).flatten()

minlength= min(log_spec_1.size, log_spec_2.size)
logspec1=logspec1[0:minlength]
log_spec_2=log_spec_2[:minlength]


sim = 1 - cosine(logspec1, log_spec_2)
percent_sim = int(sim*100)

print("\nsimilarity is approx{}%".format(percent_sim))
print("finished")
               
               
               
