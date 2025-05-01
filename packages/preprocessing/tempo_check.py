import os
import mido
from mido import MidiFile, bpm2tempo, tempo2bpm

def extract_tempos(midi_file):
    """
    MIDI 파일에서 등장하는 모든 템포 값을 BPM 단위로 반환합니다.
    중복은 제거합니다.
    """
    tempos = set()
    for msg in midi_file:
        if msg.type == 'set_tempo':
            bpm = tempo2bpm(msg.tempo)
            tempos.add(round(bpm, 2))  # 소수점 2자리로 정리
    return sorted(tempos)

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
                tempos = extract_tempos(midi_file)
                print(f"{filename}: {tempos if tempos else '[No tempo information]'}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")

# 사용 예시
directory_path = 'converted_120bpm_midis'
print_all_tempos(directory_path)
