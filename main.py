from typing import Tuple, List, TypedDict
import librosa
import soundfile as sf
import numpy as np
from math import log2, pow
from tqdm import tqdm
import crepe


class FrequencyPeriod(TypedDict):
    start_time: float
    end_time: float
    frequency: float
    key_step: int

# meow-short.wav:
# freq=3965.7969
# pitch='B7'


def load_audio(audio_path: str = "./vocals.wav", start_time: int = 0, end_time: int = 100000) -> Tuple[int, np.ndarray]:
    # AudioMetaData(sample_rate=44100, num_frames=109368, num_channels=2,
    # bits_per_sample=16, encoding=PCM_S)
    # metadata = torchaudio.info(audio_path)
    # print(f"{vars(metadata)=}")

    waveform, sample_rate = librosa.load(audio_path)
    return sample_rate, waveform[start_time * sample_rate:end_time * sample_rate]


def get_frequencies(y: np.ndarray, sr: int) -> List[FrequencyPeriod]:
    # hop_length = 512 * 12
    hop_length = 512 * 2
    audio_stft = np.abs(librosa.stft(y, hop_length=hop_length, n_fft=2048))
    pitches, magnitudes = librosa.core.piptrack(
        S=audio_stft, sr=sr, hop_length=hop_length,
        fmin=100, fmax=400, threshold=1, ref=np.mean)
    time_pitch = []
    for t in range(pitches.shape[1]):
        # Find the index of the maximum magnitude
        max_magnitude_index = magnitudes[:, t].argmax()
        # Get the corresponding pitch
        pitch = pitches[max_magnitude_index, t]

        magnitude_value = magnitudes[:, t][max_magnitude_index]
        start_time = t * hop_length / sr
        end_time = (t+1) * hop_length / sr
        # Ignore pitch values of 0, as they indicate silence or unvoiced frames
        if pitch <= 0:
            continue
        key_step = freq_to_pitch(pitch)
        close_to_last = False
        if len(time_pitch) > 0 and magnitude_value > 1.5:
            last_step = time_pitch[len(time_pitch)-1]["key_step"]
            current_step = key_step
            close_to_last = abs(last_step - current_step) <= 1
        if magnitude_value > 2 or close_to_last:
            time_pitch.append(
                {"start_time": start_time,
                    "end_time": end_time,
                    "frequency": pitch,
                    "key_step": key_step})

    # print('\n'.join([f"{item['key_step']}: {item['start_time']:.2f}-{item['end_time']:.2f}s" for item in time_pitch]))
    # print(f"{time_pitch=}")
    # print(f"{pitches.shape=}, {magnitudes.shape=}")
    return time_pitch


A4 = 440
C0 = A4*pow(2, -4.75)
name = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def freq_to_pitch(freq: float) -> int:
    h = 12*log2(freq/C0)
    return h


def freq_to_pitch_name(freq: float) -> str:
    h = round(freq_to_pitch(freq))
    octave = h // 12
    n = h % 12
    return name[n] + str(octave)


def group_frequencies(
        frequencies: List[FrequencyPeriod]) -> List[FrequencyPeriod]:
    grouped_frequencies = []
    # last_pitch = ""
    last_pitch = 0
    for frequency_period in frequencies:
        # pitch = freq_to_pitch_name(frequency_period["frequency"])
        pitch_step = frequency_period["key_step"]
        # print(f'{frequency_period["end_time"] - frequency_period["start_time"]=}, {pitch=},{last_pitch=}')
        # if pitch == last_pitch:
        if abs(last_pitch - pitch_step) <= 1.5:
            grouped_frequencies[len(grouped_frequencies)-1]["end_time"] = frequency_period["end_time"]
        else:
            grouped_frequencies.append(frequency_period)
            last_pitch = pitch_step

    filtered_frequencies = []
    for frequency_period in grouped_frequencies:
        if frequency_period["end_time"] - frequency_period["start_time"] < 0.07:
            continue
        filtered_frequencies.append(frequency_period)
    print('\n'.join([f"{item['key_step']:.1f}: {item['start_time']:.2f}-{item['end_time']:.2f}s" for item in filtered_frequencies]))
    return filtered_frequencies


def create_miauws(audio_shape, audio_sr, frequency_periods: List[FrequencyPeriod]):
    miauw_sr, miauw_audio = load_audio("./meow-short.wav", end_time=100000)
    miauw_length = miauw_audio.shape[0] / miauw_sr
    miauw_frequency = 3965.7969
    miauw_steps = freq_to_pitch(miauw_frequency) - 43

    wav_data = np.zeros(int(audio_shape[0] * miauw_sr / audio_sr), dtype=np.float32)

    for i, frequency_period in tqdm(enumerate(frequency_periods)):
        audio_length = frequency_period["end_time"] - frequency_period["start_time"]
        if i < len(frequency_periods) - 1:
            time_diff = frequency_periods[i + 1]["start_time"] - frequency_period["end_time"]
            if time_diff > 0.05:
                audio_length = max(audio_length, 0.5)
            else:
                audio_length = max(audio_length, 0.2)
        stretch_factor = miauw_length / audio_length
        steps = frequency_period["key_step"]
        pitch_shifted = librosa.effects.pitch_shift(miauw_audio, sr=miauw_sr, n_steps=steps - miauw_steps)
        pitch_shifted_stretched = librosa.effects.time_stretch(pitch_shifted, rate=stretch_factor)
        start_sample = int(frequency_period["start_time"] * miauw_sr)
        # print(f"{pitch_shifted_stretched.shape=}, {start_sample=}, {wav_data.shape[0] -start_sample=}")
        # print(f"{stretch_factor=}, {pitch_shifted.shape=}, {pitch_shifted_stretched.shape=}, {start_sample=}, {wav_data.shape=}")
        max_duration = wav_data.shape[0] - (start_sample + pitch_shifted_stretched.shape[0])
        pitch_shifted_stretched = pitch_shifted_stretched[:max_duration]
        wav_data[start_sample:start_sample + pitch_shifted_stretched.shape[0]] += pitch_shifted_stretched
    return wav_data


def merge(miauw_path: str, instrumental_path: str, output_path: str, start_time: int = 0, end_time: int = 1000000):
    sr, x = load_audio(miauw_path, end_time=end_time)
    sr, y = load_audio(instrumental_path, start_time=start_time, end_time=end_time)
    z = x + y
    sf.write(output_path, z, sr, 'PCM_24')


def main():
    # start_time = 7
    # end_time = 300
    # sample_rate, audio = load_audio(start_time=start_time, end_time=end_time)
    # frequencies = get_frequencies(audio, sample_rate)
    # frequencies = group_frequencies(frequencies)
    # miauwed = create_miauws(audio.shape, sample_rate, frequencies)
    # sf.write("miauw.wav", miauwed, sample_rate, 'PCM_24')

    # merge(miauw_path="miauw.wav", instrumental_path="instrumental.wav", output_path="merged.wav", start_time=start_time, end_time=end_time)
    sr, y = load_audio(end_time=16)
    result = crepe.predict(y, sr)
    print(result)


if __name__ == "__main__":
    main()
