import os
import mido
from mido import MidiFile, bpm2tempo, tempo2bpm

'''
파일 BPM 출력
'''
def print_tempos(midi_path):
    midi_file = MidiFile(midi_path)
    tempos = set()
    for msg in midi_file:
        if msg.type == 'set_tempo':
            bpm = tempo2bpm(msg.tempo)
            tempos.add(round(bpm, 2))
    print(f'{midi_file}: {tempos if tempos else "[No tempo information]"}')
'''
디렉토리 내 모든 midi 파일의 BPM 출력
'''
def print_all_tempos(directory_path):
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if not filename.endswith('.midi'):
            continue

        else:
            midi_path = os.path.join(directory_path, filename)
            try:
                print_tempos(midi_path)
            except Exception as e:
                print(f'Error processing {filename}: {e}')
