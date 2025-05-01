import os
import mido
from mido import MidiFile

def get_time_signature(midi_file):
    """
    MIDI 파일에서 박자표 정보를 추출하는 함수.
    4/4 박자이면 True, 아니면 False 반환.
    """
    for msg in midi_file:
        if msg.type == 'time_signature':
            numerator = msg.numerator  # 분자 (박자의 개수)
            if numerator == 4:
                return True  # 4/4 박자
            else:
                return False  # 4/4가 아닌 박자
    return False  # 기본값으로 4/4가 아닌 박자

def analyze_midi_directory(directory_path):
    """
    주어진 디렉토리 내 모든 MIDI 파일을 분석하여 4/4 박자와 4/4가 아닌 박자의 개수 및 비율을 출력하는 함수.
    """
    total_files = 0
    four_four_count = 0
    non_four_four_count = 0
    
    # 디렉토리 내 모든 파일 검사
    for filename in os.listdir(directory_path):
        if filename.endswith(".midi"):
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
                print(f"Error processing {filename}: {e}")
    
    if total_files > 0:
        ratio_four_four = four_four_count / total_files * 100
        print(f"4/4 박자 파일 개수: {four_four_count}")
        print(f"4/4가 아닌 박자 파일 개수: {non_four_four_count}")
        print(f"전체 파일 중 4/4 박자의 비율: {ratio_four_four:.2f}%")
    else:
        print("디렉토리에 MIDI 파일이 없습니다.")

# 사용 예시
print(os.getcwd())
directory_path = os.path.join('midi', 'train_')
analyze_midi_directory(directory_path)
