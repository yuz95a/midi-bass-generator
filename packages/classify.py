import os
import shutil
import pretty_midi
import glob

def has_bass_track(midi_file):
    """
    midi 파일에 베이스 트랙이 있는지 확인하는 함수
    """
    try:
        # midi 파일 로드
        midi_data = pretty_midi.PrettyMIDI(midi_file)
        
        # 각 악기 트랙을 확인
        for instrument in midi_data.instruments:
            # 악기 이름에 'bass'가 포함되어 있거나 프로그램 번호가 베이스에 해당하는 경우(32-39)
            if ('bass' in instrument.name.lower() or 
                (32 <= instrument.program <= 39)):
                return True
        
        return False
    except Exception as e:
        print(f"Error processing {midi_file}: {e}")
        return False

def copy_midi_with_bass(source_dir, target_dir, target_dir_no):
    """
    베이스 트랙이 있는 midi 파일만 대상 디렉토리로 복사
    """
    # 대상 디렉토리가 없으면 생성
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    
    if not os.path.exists(target_dir_no):
        os.makedirs(target_dir_no)
    
    # 소스 디렉토리에서 모든 midi 파일 경로 가져오기
    midi_files = glob.glob(os.path.join(source_dir, "**", "*.midi"), recursive=True)
    
    copied_count = 0
    total_count = len(midi_files)
    
    print(f"총 {total_count}개의 MIDI 파일을 처리합니다...")
    
    for midi_file in midi_files:
        # 파일 이름만 추출
        file_name = os.path.basename(midi_file)
        if has_bass_track(midi_file):
            # 대상 경로 생성
            target_path = os.path.join(target_dir, file_name)
            # 파일 복사
            shutil.copy2(midi_file, target_path)
            copied_count += 1
            print(f"복사됨: {file_name}")
        else:
            # 대상 경로 생성
            target_path = os.path.join(target_dir_no, file_name)
            # 파일 복사
            shutil.copy2(midi_file, target_path)
            print(f"복사됨: {file_name}")
    
    print(f"처리 완료: {total_count}개 중 {copied_count}개의 베이스 트랙이 있는 MIDI 파일을 복사했습니다.")
    print(f"처리 완료: {total_count}개 중 {total_count - copied_count}개의 베이스 트랙이 없는 MIDI 파일을 복사했습니다.")



def main(midi_dir, target_dir, target_dir_no):
    copy_midi_with_bass(midi_dir, target_dir, target_dir_no)

if __name__ == '__main__':
    main('./XMIDI/pop', './XMIDIs/pop', './XMIDIs/pop_no_bass')