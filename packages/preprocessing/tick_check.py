import os
import mido
from mido import MidiFile

def print_all_tempos(directory_path):
    """
    디렉토리 내 모든 MIDI 파일에 대해 템포 정보를 출력합니다.
    """
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if not filename.endswith(".midi"):
            continue

        # if i % 100 != 0:
        #     continue

        if filename.endswith(".midi"):
            midi_path = os.path.join(directory_path, filename)
            try:
                midi_file = MidiFile(midi_path)
                ticks_per_beat = midi_file.ticks_per_beat
                print(f"{filename}: {ticks_per_beat if ticks_per_beat else '[No tempo information]'}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 사용 예시
directory_path = os.path.join('midi', 'train_')
print_all_tempos(directory_path)
