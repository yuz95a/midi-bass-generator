import os
import mido
from mido import MidiFile

'''
4/4 박자면 True 아니면 False
'''
def get_time_signature(midi_file):
    for msg in midi_file:
        if msg.type == 'time_signature':
            numerator = msg.numerator
            if numerator == 4:
                return True
            else:
                return False
    return False

'''
디렉토리 내 모든 midi 파일 검사
4/4 박자와 4/4가 아닌 박자의 개수 및 비율 출력
'''
def print_all_time_signature(directory_path):
    total_files = 0
    four_four_count = 0
    non_four_four_count = 0
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.midi'):
            total_files += 1
            midi_path = os.path.join(directory_path, filename)
            
            try:
                midi_file = MidiFile(midi_path)
                if get_time_signature(midi_file):
                    four_four_count += 1
                    print(True)
                else:
                    non_four_four_count += 1
                    print(False)
            except Exception as e:
                print(f'Error processing {filename}: {e}')
    
    if total_files > 0:
        ratio_four_four = four_four_count / total_files * 100
        print(f'4/4 박자 파일 개수: {four_four_count}')
        print(f'4/4가 아닌 박자 파일 개수: {non_four_four_count}')
        print(f'전체 파일 중 4/4 박자의 비율: {ratio_four_four:.2f}%')
    else:
        print('디렉토리에 MIDI 파일이 없습니다.')
