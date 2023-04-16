import torchaudio
from typing import Tuple, List
import torch
import librosa
import numpy as np
from math import log2, pow


def load_audio(audio_path: str = "./vocals.wav") -> Tuple[int, np.ndarray]:
    # AudioMetaData(sample_rate=44100, num_frames=109368, num_channels=2, bits_per_sample=16, encoding=PCM_S)
    metadata = torchaudio.info(audio_path)
    print(f"{vars(metadata)=}")
    
    waveform, sample_rate = librosa.load(audio_path)
    print(f"{sample_rate=}")
    print(f"{waveform.shape=}")
    return sample_rate, waveform
    
    
def get_pitch(waveform: np.ndarray, start_sample_time: int, end_sample_time: int):
    # Compute the Short-Time Fourier Transform (STFT)
    stft = librosa.stft(waveform[start_sample_time:end_sample_time])

    # Compute the magnitude spectrogram
    magnitude = np.abs(stft)

    # Find the frequency index with the maximum magnitude for each time frame
    peak_indices = np.argmax(magnitude, axis=0)

    # Convert frequency indices to frequencies in Hz
    frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=stft.shape[0])
    peak_frequencies = frequencies[peak_indices]

    print(peak_frequencies)
    return peak_frequencies


def get_onsets(waveform: np.ndarray, sample_rate: int) -> List[int]:
    # Perform onset detection
    onset_frames = librosa.onset.onset_detect(y=waveform, sr=sample_rate)

    print(f"{onset_frames=}")
    return onset_frames
    
    
def time_to_samples(time: float, sample_rate: int) -> int:
    return int(time * sample_rate)
    
    
def get_frequencies(y: np.ndarray, sr: int) -> List[Tuple[float, float, float]]:
    audio_stft = np.abs(librosa.stft(y[0:sr*20]))
    hop_length = 512
    pitches, magnitudes = librosa.core.piptrack(S=audio_stft, sr=sr, hop_length=hop_length) # , fmin=75, fmax=1600
    time_pitch = []
    for t in range(pitches.shape[1]):
        max_magnitude_index = magnitudes[:, t].argmax()  # Find the index of the maximum magnitude
        pitch = pitches[max_magnitude_index, t]  # Get the corresponding pitch

        magnitude_value = magnitudes[:, t][max_magnitude_index]
        start_time = t * hop_length / sr
        end_time = (t+1) * hop_length / sr
        # print(f"{magnitude_value=}, {time=}")
        if pitch > 0 and magnitude_value>10:  # Ignore pitch values of 0, as they indicate silence or unvoiced frames
            time_pitch.append((start_time, end_time, pitch))
    print(f"{time_pitch=}")
    print(f"{pitches.shape=}, {magnitudes.shape=}")
    return pitches


A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    
def freq_to_pitch(freq: float):
    h = round(12*log2(freq/C0))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)


def group_frequencies(frequencies: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    grouped_frequencies = []
    last_pitch = ""
    for start_time, end_time, frequency in frequencies:
        pitch = freq_to_pitch(frequency)
        if pitch == last_pitch:
            grouped_frequencies[len(grouped_frequencies)-1][1] = end_time
        else:
            grouped_frequencies.append((start_time, end_time, frequency))
            last_pitch = pitch
    return grouped_frequencies


def create_miauws(pitches):
    return []


def main():
    print("Floris")
    sample_rate, audio = load_audio()
    frequencies = get_frequencies(audio, sample_rate)
    frequencies = group_frequencies(frequencies)
    print(f"{frequencies=}")
    # pitches = get_pitches(sample_rate, audio)
    # miauws = create_miauws(pitches)


# stft = librosa.stft(y)  # Compute the STFT
# magnitude_spectrogram = np.abs(stft)  # Obtain the magnitude spectrogram
#
# pitches, magnitudes = librosa.piptrack(S=magnitude_spectrogram, sr=sr)
#
# time_pitch = []

# for t in range(pitches.shape[1]):
#     max_magnitude_index = magnitudes[:, t].argmax()  # Find the index of the maximum magnitude
#     pitch = pitches[max_magnitude_index, t]  # Get the corresponding pitch

#     if pitch > 0:  # Ignore pitch values of 0, as they indicate silence or unvoiced frames
#         time_pitch.append((t / sr, pitch))

# print(time_pitch)

if __name__=="__main__":
    main()