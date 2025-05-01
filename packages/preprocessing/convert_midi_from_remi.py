import os
import pickle
import pretty_midi
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import re

class REMIToMIDIConverter:
    """REMI 토큰을 MIDI 파일로 변환하는 클래스"""
    
    def __init__(self, resolution=24):
        """
        초기화
        Args:
            resolution: 분해능 (4분 음표 기준 틱 수)
        """
        self.resolution = resolution
        self.current_tempo = 120.0  # 기본 템포
        
        # 토큰 타입 정규식 패턴
        self.token_patterns = {
            'Bar': r'^Bar$',
            'Position': r'^Position_(\d+)$',
            'Note_On': r'^Note_On_(\d+)$',
            'Note_Duration': r'^Note_Duration_(\d+)$',
            'Tempo': r'^Tempo_(\d+(?:\.\d+)?)$',
            'Program': r'^Program_(\d+)$',
            'Drum_Hit': r'^Drum_Hit$',
        }
        
        # 토큰 타입 정규식 컴파일
        self.token_regexes = {name: re.compile(pattern) for name, pattern in self.token_patterns.items()}
    
    def parse_token(self, token: str) -> Tuple[str, Optional[Union[int, float]]]:
        """
        토큰 문자열을 타입과 값으로 파싱
        Args:
            token: REMI 토큰 문자열
        Returns:
            토큰 타입과 값(있을 경우)의 튜플
        """
        for token_type, regex in self.token_regexes.items():
            match = regex.match(token)
            if match:
                if len(match.groups()) > 0:
                    # 값이 있는 경우 (Position, Note_On 등)
                    value = match.group(1)
                    try:
                        if '.' in value:
                            return token_type, float(value)
                        else:
                            return token_type, int(value)
                    except ValueError:
                        return token_type, value
                else:
                    # 값이 없는 경우 (Bar, Drum_Hit 등)
                    return token_type, None
        
        # 특수 토큰 처리
        if token in ['BASS_DRUM_SEPARATOR', 'DRUM_OTHER_SEPARATOR', 'TRACK_SEPARATOR']:
            return 'Separator', token
        
        # 매치되지 않는 경우
        return 'Unknown', None
    
    def create_midi_from_tokens(self, tokens_dict: Dict[str, List[str]], output_path: str) -> Optional[pretty_midi.PrettyMIDI]:
        """
        REMI 토큰을 MIDI 파일로 변환
        Args:
            tokens_dict: 'bass_tokens', 'drum_tokens', 'other_tokens'를 담은 딕셔너리
            output_path: 출력 MIDI 파일 경로
        Returns:
            생성된 PrettyMIDI 객체 또는 실패 시 None
        """
        try:
            midi = pretty_midi.PrettyMIDI(initial_tempo=self.current_tempo)
            
            # 베이스 트랙 처리
            if 'bass_tokens' in tokens_dict and tokens_dict['bass_tokens']:
                bass_instrument = self.tokens_to_instrument(tokens_dict['bass_tokens'], is_bass=True)
                if bass_instrument:
                    midi.instruments.append(bass_instrument)
            
            # 드럼 트랙 처리
            if 'drum_tokens' in tokens_dict and tokens_dict['drum_tokens']:
                drum_instrument = self.tokens_to_instrument(tokens_dict['drum_tokens'], is_drum=True)
                if drum_instrument:
                    midi.instruments.append(drum_instrument)
            
            # 기타 트랙 처리
            if 'other_tokens' in tokens_dict and tokens_dict['other_tokens']:
                other_instrument = self.tokens_to_instrument(tokens_dict['other_tokens'])
                if other_instrument:
                    midi.instruments.append(other_instrument)
            
            # 결합된 토큰 처리 (tokens_dict에 'combined_tokens'만 있는 경우)
            if ('combined_tokens' in tokens_dict and tokens_dict['combined_tokens'] and 
                'bass_tokens' not in tokens_dict and 'drum_tokens' not in tokens_dict and 'other_tokens' not in tokens_dict):
                # 결합된 토큰을 개별 트랙으로 분리
                bass_tokens, drum_tokens, other_tokens = self.split_combined_tokens(tokens_dict['combined_tokens'])
                
                bass_instrument = self.tokens_to_instrument(bass_tokens, is_bass=True)
                if bass_instrument:
                    midi.instruments.append(bass_instrument)
                
                drum_instrument = self.tokens_to_instrument(drum_tokens, is_drum=True)
                if drum_instrument:
                    midi.instruments.append(drum_instrument)
                
                other_instrument = self.tokens_to_instrument(other_tokens)
                if other_instrument:
                    midi.instruments.append(other_instrument)
            
            # MIDI 파일 저장
            midi.write(output_path)
            print(f"MIDI file saved to {output_path}")
            
            return midi
        
        except Exception as e:
            print(f"Error creating MIDI from tokens: {e}")
            return None
    
    def split_combined_tokens(self, combined_tokens: List[str]) -> Tuple[List[str], List[str], List[str]]:
        """
        결합된 토큰을 베이스, 드럼, 기타 트랙으로 분리
        Args:
            combined_tokens: 결합된 토큰 리스트
        Returns:
            (베이스 토큰, 드럼 토큰, 기타 토큰) 튜플
        """
        bass_tokens = []
        drum_tokens = []
        other_tokens = []
        
        current_section = 'bass'
        
        for token in combined_tokens:
            if token == 'BASS_DRUM_SEPARATOR':
                current_section = 'drum'
            elif token == 'DRUM_OTHER_SEPARATOR':
                current_section = 'other'
            elif token == 'TRACK_SEPARATOR':
                # 이전 포맷 지원
                if current_section == 'bass':
                    current_section = 'other'
                else:
                    current_section = 'bass'
            else:
                if current_section == 'bass':
                    bass_tokens.append(token)
                elif current_section == 'drum':
                    drum_tokens.append(token)
                else:
                    other_tokens.append(token)
        
        return bass_tokens, drum_tokens, other_tokens
    
    def tokens_to_instrument(self, tokens: List[str], is_bass: bool = False, is_drum: bool = False) -> Optional[pretty_midi.Instrument]:
        """
        REMI 토큰을 MIDI 악기로 변환
        Args:
            tokens: REMI 토큰 리스트
            is_bass: 베이스 트랙 여부
            is_drum: 드럼 트랙 여부
        Returns:
            처리된 MIDI 악기 객체
        """
        if not tokens:
            return None
        
        # 기본 프로그램 설정
        program = 33 if is_bass else 0  # 33: 핑거베이스, 0: 어쿠스틱 그랜드 피아노
        
        # 드럼인 경우 설정
        instrument = pretty_midi.Instrument(program=program, is_drum=is_drum)
        if is_drum:
            instrument.name = "Drums"
        elif is_bass:
            instrument.name = "Bass"
        else:
            instrument.name = "Other"
        
        current_tick = 0
        bar_count = 0
        ticks_per_bar = 4 * self.resolution  # 4/4 박자 기준
        
        current_position = 0
        current_notes = []  # (pitch, velocity, start_tick) 튜플 리스트
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            token_type, value = self.parse_token(token)
            
            if token_type == 'Bar':
                bar_count += 1
                current_tick = bar_count * ticks_per_bar
                current_position = 0
            
            elif token_type == 'Position':
                current_position = value
                # 바 시작부터의 틱 계산
                current_tick = (bar_count * ticks_per_bar) + current_position
            
            elif token_type == 'Program':
                if not is_drum:  # 드럼이 아닌 경우만 프로그램 변경
                    program = value
                    instrument.program = program
            
            elif token_type == 'Note_On':
                # 다음 토큰이 Duration인지 확인
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    next_type, next_value = self.parse_token(next_token)
                    
                    if next_type == 'Note_Duration':
                        pitch = value
                        duration_ticks = next_value
                        start_time = self.tick_to_time(current_tick)
                        end_time = self.tick_to_time(current_tick + duration_ticks)
                        
                        # 노트 생성
                        note = pretty_midi.Note(
                            velocity=100,  # 기본 베로시티
                            pitch=pitch,
                            start=start_time,
                            end=end_time
                        )
                        instrument.notes.append(note)
                        
                        i += 1  # Duration 토큰 건너뛰기
            
            elif token_type == 'Drum_Hit':
                # 다음 토큰이 Duration인지 확인
                if i + 1 < len(tokens):
                    next_token = tokens[i + 1]
                    next_type, next_value = self.parse_token(next_token)
                    
                    if next_type == 'Note_Duration':
                        duration_ticks = next_value
                        start_time = self.tick_to_time(current_tick)
                        end_time = self.tick_to_time(current_tick + duration_ticks)
                        
                        # 드럼 히트 - 여러 드럼 음높이를 사용해 리듬 표현
                        drum_pitches = [36, 38, 42]  # 베이스 드럼, 스네어, 하이햇
                        
                        # 베이스 드럼 (1, 5 박)
                        if current_position % (self.resolution) == 0 or current_position % (self.resolution) == (self.resolution//2)*2:
                            note = pretty_midi.Note(
                                velocity=100,
                                pitch=36,  # 베이스 드럼
                                start=start_time,
                                end=end_time
                            )
                            instrument.notes.append(note)
                        
                        # 스네어 (2, 4 박)
                        elif current_position % (self.resolution) == self.resolution or current_position % (self.resolution * 2) == self.resolution * 3:
                            note = pretty_midi.Note(
                                velocity=100,
                                pitch=38,  # 스네어
                                start=start_time,
                                end=end_time
                            )
                            instrument.notes.append(note)
                        
                        # 그 외: 하이햇
                        else:
                            note = pretty_midi.Note(
                                velocity=90,
                                pitch=42,  # 클로즈드 하이햇
                                start=start_time,
                                end=end_time
                            )
                            instrument.notes.append(note)
                        
                        i += 1  # Duration 토큰 건너뛰기
            
            elif token_type == 'Tempo':
                self.current_tempo = value
            
            i += 1
        
        return instrument
    
    def tick_to_time(self, tick: int) -> float:
        """
        틱을 시간(초)으로 변환
        Args:
            tick: MIDI 틱
        Returns:
            시간(초)
        """
        beats_per_second = self.current_tempo / 60.0
        seconds_per_tick = 1.0 / (beats_per_second * self.resolution)
        return tick * seconds_per_tick
    
    def convert_file(self, remi_path: str, output_path: str = None) -> Optional[pretty_midi.PrettyMIDI]:
        """
        REMI 파일을 MIDI로 변환
        Args:
            remi_path: REMI 픽클 파일 경로
            output_path: 출력 MIDI 파일 경로
        Returns:
            생성된 PrettyMIDI 객체 또는 실패 시 None
        """
        try:
            # REMI 파일 로드
            with open(remi_path, 'rb') as f:
                remi_data = pickle.load(f)
            
            if output_path is None:
                # 출력 경로가 지정되지 않은 경우 기본 경로 생성
                base_path = os.path.splitext(remi_path)[0]
                output_path = f"{base_path}.mid"
            
            # MIDI 생성
            return self.create_midi_from_tokens(remi_data, output_path)
            
        except Exception as e:
            print(f"Error converting REMI file to MIDI: {e}")
            return None

def process_remi_directory(input_dir: str, output_dir: str, resolution: int = 24):
    """
    디렉토리 내 모든 REMI 파일을 MIDI로 변환
    Args:
        input_dir: 입력 REMI 파일 디렉토리
        output_dir: 출력 MIDI 파일 디렉토리
        resolution: 분해능
    """
    converter = REMIToMIDIConverter(resolution=resolution)
    
    os.makedirs(output_dir, exist_ok=True)
    
    count_success = 0
    count_failed = 0
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pkl'):
            remi_path = os.path.join(input_dir, file_name)
            midi_name = os.path.splitext(file_name)[0] + '.mid'
            output_path = os.path.join(output_dir, midi_name)
            
            print(f"Converting {file_name} to MIDI...")
            result = converter.convert_file(remi_path, output_path)
            
            if result:
                count_success += 1
            else:
                count_failed += 1
    
    print(f"\nConversion complete. Success: {count_success}, Failed: {count_failed}")

# 사용 예시
if __name__ == "__main__":
    # 단일 파일 변환
    converter = REMIToMIDIConverter()
    converter.convert_file("XMIDI_angry_rock_H1J1B70Z_remi.pkl", "output.mid")
    
    # 디렉토리 내 모든 파일 변환
    # process_remi_directory("remi_output", "midi_output")