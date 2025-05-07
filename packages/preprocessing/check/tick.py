import os
import mido
from mido import MidiFile

'''
파일 ticks_per_beat 출력
'''
def print_ticks(midi_path):
    midi_file = MidiFile(midi_path)
    ticks_per_beat = midi_file.ticks_per_beat
    print(f'{filename}: {ticks_per_beat if ticks_per_beat else "[No tempo information]"}')
'''
디렉토리 내 모든 midi 파일의 ticks_per_beat 출력
'''
def print_all_ticks(directory_path):
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if not filename.endswith('.midi'):
            continue

        else:
            midi_path = os.path.join(directory_path, filename)
            try:
                print_ticks(midi_path)
            except Exception as e:
                print(f'Error processing {filename}: {e}')
