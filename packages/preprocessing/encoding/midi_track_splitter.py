import os
import argparse
import pretty_midi
from pathlib import Path

def identify_bass_track(pm):
    """베이스 트랙을 식별하는 함수
    
    베이스 트랙 식별 방법:
    1. 악기 이름에 'bass'가 포함되어 있는지 확인
    2. 악기 프로그램이 베이스 음색(32-39)인지 확인
    
    위 조건 중 하나라도 충족하는 첫 번째 트랙을 베이스 트랙으로 선택
    """
    for i, instrument in enumerate(pm.instruments):
        # 드럼 트랙은 건너뛰기
        if instrument.is_drum:
            continue
            
        # 악기 이름 확인
        has_bass_name = False
        if instrument.name and 'bass' in instrument.name.lower():
            has_bass_name = True
        
        # 베이스 프로그램 확인 (베이스 음색: 32-39)
        has_bass_program = 32 <= instrument.program <= 39
        
        # 노트가 있고, 이름이나 프로그램이 베이스인 경우 선택
        if instrument.notes and (has_bass_name or has_bass_program):
            return i
    
    # 베이스 트랙을 찾지 못한 경우 None 반환
    return None

def split_midi_file(input_path, bass_output_path, other_output_path):
    """MIDI 파일을 베이스와 나머지 트랙으로 분리"""
    try:
        # MIDI 파일 로드
        pm = pretty_midi.PrettyMIDI(input_path)
        
        # 베이스 트랙 식별
        bass_track_index = identify_bass_track(pm)
        
        if bass_track_index is None:
            print(f"베이스 트랙을 찾을 수 없습니다: {input_path}")
            return False
        
        # 베이스 트랙만 있는 MIDI 파일 생성
        bass_pm = pretty_midi.PrettyMIDI(initial_tempo=pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1] else 120)
        bass_pm.time_signature_changes = pm.time_signature_changes
        bass_pm.key_signature_changes = pm.key_signature_changes
        
        # 베이스 트랙 제외한 MIDI 파일 생성
        other_pm = pretty_midi.PrettyMIDI(initial_tempo=pm.get_tempo_changes()[1][0] if pm.get_tempo_changes()[1] else 120)
        other_pm.time_signature_changes = pm.time_signature_changes
        other_pm.key_signature_changes = pm.key_signature_changes
        
        # 악기 분리
        for i, instrument in enumerate(pm.instruments):
            if i == bass_track_index:
                # 베이스 트랙은 베이스 MIDI에 추가
                bass_pm.instruments.append(instrument)
            else:
                # 나머지 트랙은 other MIDI에 추가
                other_pm.instruments.append(instrument)
        
        # 파일 저장
        bass_pm.write(bass_output_path)
        other_pm.write(other_output_path)
        
        return True
    
    except Exception as e:
        print(f"오류 발생: {input_path} - {str(e)}")
        return False

def process_directory(origin_dir, bass_dir, other_dir):
    """디렉토리 내 모든 MIDI 파일을 처리"""
    # 출력 디렉토리 생성
    os.makedirs(bass_dir, exist_ok=True)
    os.makedirs(other_dir, exist_ok=True)
    
    success_count = 0
    error_count = 0
    
    # 모든 MIDI 파일 처리
    for file_path in Path(origin_dir).glob('**/*.mid*'):
        relative_path = file_path.relative_to(origin_dir)
        
        # 출력 파일 경로 생성
        bass_output_path = Path(bass_dir) / relative_path
        other_output_path = Path(other_dir) / relative_path
        
        # 출력 디렉토리 생성
        bass_output_path.parent.mkdir(parents=True, exist_ok=True)
        other_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"처리 중: {file_path}")
        
        # 파일 분리
        success = split_midi_file(str(file_path), str(bass_output_path), str(other_output_path))
        
        if success:
            success_count += 1
        else:
            error_count += 1
    
    print(f"\n처리 완료: 성공 {success_count}, 실패 {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MIDI 파일에서 베이스 트랙을 분리합니다.')
    parser.add_argument('--origin', default='origin', help='원본 MIDI 파일이 있는 디렉토리')
    parser.add_argument('--bass', default='bass', help='베이스 트랙만 있는 MIDI 파일을 저장할 디렉토리')
    parser.add_argument('--other', default='other', help='베이스 트랙을 제외한 MIDI 파일을 저장할 디렉토리')
    
    args = parser.parse_args()
    
    process_directory(args.origin, args.bass, args.other)