import os
import mido
from mido import MidiFile

'''
midi 파일에 set_tempo 메시지가 여러 개 있는지 확인
템포가 한 번 이상 바뀌면 True, 아니면 False 반환
'''
def has_multiple_tempos(midi_file):
    tempo_changes = 0
    for msg in midi_file:
        if msg.type == 'set_tempo':
            tempo_changes += 1
            if tempo_changes > 1:
                print(False)
                return True  # 템포가 두 번 이상 설정됨 (변화 있음)
    if tempo_changes == 1:
        print(True)
        return False  # 템포 변화 없음
    else:
        print('ERROR')
        return True
'''
디렉토리 내 모든 midi 파일을 분석하여
템포가 바뀌는 파일과 바뀌지 않는 파일 수 및 비율 출력
'''
def analyze_tempo_changes(directory_path):
    total_files = 0
    changed_tempo_count = 0
    unchanged_tempo_count = 0

    for filename in os.listdir(directory_path):
        if filename.endswith('.midi'):
            total_files += 1
            midi_path = os.path.join(directory_path, filename)

            try:
                midi_file = MidiFile(midi_path)
                if has_multiple_tempos(midi_file):
                    changed_tempo_count += 1
                else:
                    unchanged_tempo_count += 1
            except Exception as e:
                print(f'Error processing {filename}: {e}')

    if total_files > 0:
        ratio_unchanged = unchanged_tempo_count / total_files * 100
        print(f'템포가 바뀌지 않는 파일 개수: {unchanged_tempo_count}')
        print(f'템포가 바뀌는 파일 개수: {changed_tempo_count}')
        print(f'전체 중 템포가 바뀌지 않는 파일 비율: {ratio_unchanged:.2f}%')
    else:
        print('디렉토리에 MIDI 파일이 없습니다.')
