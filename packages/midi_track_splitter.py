import os
import mido
import argparse
from pathlib import Path

def identify_bass_track(mid):
    """베이스 트랙을 식별하는 함수
    
    베이스 트랙을 식별하는 몇 가지 방법:
    1. 트랙 이름에 'bass'가 포함되어 있는지 확인
    2. 프로그램 변경 메시지에서 베이스 음색(32-39)인지 확인
    3. 노트의 평균 피치가 낮은 범위(약 30-60)에 있는지 확인
    """
    bass_candidates = []
    
    for i, track in enumerate(mid.tracks):
        # 트랙 이름 확인
        track_name = None
        is_bass_name = False
        notes = []
        has_bass_program = False
        
        for msg in track:
            # 트랙 이름 메시지 찾기
            if msg.type == 'track_name':
                track_name = msg.name.lower()
                if 'bass' in track_name:
                    is_bass_name = True
            
            # 프로그램 변경 메시지 확인 (베이스 음색: 32-39)
            elif msg.type == 'program_change' and 32 <= msg.program <= 39:
                has_bass_program = True
            
            # 노트 수집
            elif msg.type == 'note_on' and msg.velocity > 0:
                notes.append(msg.note)
        
        # 노트가 있는 경우만 평가
        if notes:
            avg_pitch = sum(notes) / len(notes)
            # 점수 계산 (각 특성에 가중치 부여)
            score = 0
            if is_bass_name:
                score += 3  # 이름에 'bass'가 있으면 높은 가중치
            if has_bass_program:
                score += 2  # 베이스 프로그램이면 중간 가중치
            if 30 <= avg_pitch <= 60:
                score += 1  # 낮은 피치 범위면 낮은 가중치
            
            if score > 0:  # 점수가 있는 경우만 후보로 등록
                bass_candidates.append((i, score, avg_pitch, track_name))
    
    # 후보가 없으면 None 반환
    if not bass_candidates:
        return None
    
    # 점수가 가장 높은 트랙 선택 (같은 점수면 평균 피치가 낮은 것 선택)
    bass_candidates.sort(key=lambda x: (-x[1], x[2]))
    return bass_candidates[0][0]

def split_midi_file(input_path, bass_output_path, other_output_path):
    """MIDI 파일을 베이스와 나머지 트랙으로 분리"""
    try:
        mid = mido.MidiFile(input_path)
        
        # 베이스 트랙 식별
        bass_track_index = identify_bass_track(mid)
        
        if bass_track_index is None:
            print(f"베이스 트랙을 찾을 수 없습니다: {input_path}")
            return False
        
        # 베이스 트랙만 있는 MIDI 파일 생성
        bass_mid = mido.MidiFile()
        bass_mid.ticks_per_beat = mid.ticks_per_beat
        
        # 베이스 트랙 제외한 MIDI 파일 생성
        other_mid = mido.MidiFile()
        other_mid.ticks_per_beat = mid.ticks_per_beat
        
        # 메타 트랙(일반적으로 첫 번째 트랙)이 있는지 확인
        has_meta_track = len(mid.tracks) > 0 and any(msg.type == 'track_name' or msg.is_meta for msg in mid.tracks[0])
        
        for i, track in enumerate(mid.tracks):
            # 베이스 트랙일 경우
            if i == bass_track_index:
                bass_mid.tracks.append(track)
                # 메타 트랙이 아니면 첫 번째 트랙에도 추가 (메타 정보를 포함시키기 위해)
                if i != 0 and has_meta_track:
                    bass_mid.tracks.insert(0, mido.MidiTrack())
                    for msg in mid.tracks[0]:
                        if msg.is_meta:
                            bass_mid.tracks[0].append(msg)
            else:
                other_mid.tracks.append(track)
        
        # 베이스 트랙이 포함되어 있지 않다면 첫 번째 트랙에 메타 정보 추가
        if bass_track_index != 0 and has_meta_track and bass_mid.tracks and len(bass_mid.tracks) == 1:
            meta_track = mido.MidiTrack()
            for msg in mid.tracks[0]:
                if msg.is_meta:
                    meta_track.append(msg)
            bass_mid.tracks.insert(0, meta_track)
        
        # 파일 저장
        bass_mid.save(bass_output_path)
        other_mid.save(other_output_path)
        
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