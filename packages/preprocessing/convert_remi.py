import os
import pickle
import numpy as np
import pretty_midi
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


import os
import copy
from mido import MidiFile, MidiTrack, Message, merge_tracks
from fractions import Fraction

import pickle

def split_midi_by_bars_to_file(midipath, bars=8):
    try:
        midi = MidiFile(midipath)
        
        base_name = os.path.basename(midipath)
        file_name, file_ext = os.path.splitext(base_name)
        dir_path = os.path.dirname(midipath)
        
        '''
        ticks_per_bar = 480 * 4 = 1920
        마디 당 틱 수 = 비트 당 틱 수 * 마디 당 비트 수
        대부분의 DAW에서 사용하는 ticks per beat는 480
        학습에 사용할 데이터는 4/4 박자
        '''
        ticks_per_bar = 1920
        

        # 가장 길이가 긴 트랙 찾기
        # 찾아야 하는 이유: 나중에 8마디씩 분할 후에 패딩할 때 필요
        track_lengths = []

        for track in midi.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
            track_lengths.append(track_time)

        if track_lengths:
            max_index, total_ticks = max(enumerate(track_lengths), key=lambda x: x[1])
        else:
            max_index, total_ticks = None, 0
        
        # 전체 마디 수 계산
        total_bars = (total_ticks + ticks_per_bar - 1) // ticks_per_bar

        # 8마디 이하는 분할 없이 패딩
        if total_bars <= bars:
            # MIDI 복사
            new_midi = MidiFile(ticks_per_beat=480)
            for i, track in enumerate(midi.tracks):
                new_track = MidiTrack()
                new_midi.tracks.append(new_track)
                for msg in track:
                    new_track.append(msg.copy())
            
            # 패딩할 틱 계산 후 패딩
            padding_ticks = ticks_per_bar * bars - total_ticks
            if padding_ticks > 0 and len(new_midi.tracks[-1]) > 0:
                # 패딩 메시지 추가해서 패딩 진행
                new_track.append(Message('note_on', note=0, velocity=0, time=padding_ticks))
            
            output_path = os.path.join(dir_path, f"{file_name}_0{file_ext}")
            new_midi.save(output_path)
        
        # 9마디 이상은 분할
        else:
            # 몇 개의 파일로 분할할지 계산
            split_count = (total_bars + bars - 1) // bars
            
            # 각 부분별로 처리
            for split_index in range(split_count):
                new_midi = MidiFile(ticks_per_beat=480)
                
                # 시작, 종료 틱 계산
                start_tick = split_index * ticks_per_bar * bars
                end_tick = min((split_index + 1) * ticks_per_bar * bars, total_ticks)
                
                # 메타 트랙 생성
                meta_track = MidiTrack()
                new_midi.tracks.append(meta_track)
                
                # 메타 이벤트 복사 (템포, 박자표)
                for track in midi.tracks:
                    for msg in track:
                        if msg.is_meta and (msg.type == 'set_tempo' or msg.type == 'time_signature'):
                            meta_track.append(msg.copy())
                
                # 내용 트랙 생성
                for track in midi.tracks[1:]:
                    content_track = MidiTrack()
                    new_midi.tracks.append(content_track)
                    current_tick = 0
                    # 트랙 이름, 악기 종류 메시지 복사
                    for msg in track:
                        if split_index != 0:
                            if (msg.is_meta and msg.type == 'track_name') or msg.type == 'program_change':
                                content_track.append(msg.copy())

                        current_tick += msg.time

                        if start_tick <= current_tick <= end_tick:
                            if current_tick - msg.time < start_tick:
                                new_msg = msg.copy()
                                new_msg.time = current_tick - start_tick
                                content_track.append(new_msg)
                            else:
                                content_track.append(msg.copy())
                
                # 마지막 분할에서 패딩
                if split_index == split_count - 1 and end_tick < bars * ticks_per_bar * split_count:
                    padding_ticks = bars * ticks_per_bar * split_count - end_tick
                    # 패딩 메시지 추가
                    content_track.append(Message('note_on', note=0, velocity=0, time=padding_ticks))

                output_path = os.path.join(dir_path, f"{file_name}_{split_index}{file_ext}")
                new_midi.save(output_path)

    except Exception as e:
        print(f"에러 발생: {str(e)}")

def split_midi_by_bars(midipath, bars=8):
    try:
        midi = MidiFile(midipath)
        
        base_name = os.path.basename(midipath)
        file_name, file_ext = os.path.splitext(base_name)
        dir_path = os.path.dirname(midipath)
        
        '''
        ticks_per_bar = 480 * 4 = 1920
        마디 당 틱 수 = 비트 당 틱 수 * 마디 당 비트 수
        대부분의 DAW에서 사용하는 ticks per beat는 480
        학습에 사용할 데이터는 4/4 박자
        '''
        ticks_per_bar = 1920
        

        # 가장 길이가 긴 트랙 찾기
        # 찾아야 하는 이유: 나중에 8마디씩 분할 후에 패딩할 때 필요
        track_lengths = []

        # 결과
        result = []

        for track in midi.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
            track_lengths.append(track_time)

        if track_lengths:
            max_index, total_ticks = max(enumerate(track_lengths), key=lambda x: x[1])
        else:
            max_index, total_ticks = None, 0
        
        # 전체 마디 수 계산
        total_bars = (total_ticks + ticks_per_bar - 1) // ticks_per_bar

        # 8마디 이하는 분할 없이 패딩
        if total_bars <= bars:
            # MIDI 복사
            new_midi = MidiFile(ticks_per_beat=480)
            for i, track in enumerate(midi.tracks):
                new_track = MidiTrack()
                new_midi.tracks.append(new_track)
                for msg in track:
                    new_track.append(msg.copy())
            
            # 패딩할 틱 계산 후 패딩
            padding_ticks = ticks_per_bar * bars - total_ticks
            if padding_ticks > 0 and len(new_midi.tracks[-1]) > 0:
                # 패딩 메시지 추가해서 패딩 진행
                new_track.append(Message('note_on', note=0, velocity=0, time=padding_ticks))
            
            result.append(new_midi)
        
        # 9마디 이상은 분할
        else:
            # 몇 개의 파일로 분할할지 계산
            split_count = (total_bars + bars - 1) // bars
            
            # 각 부분별로 처리
            for split_index in range(split_count):
                new_midi = MidiFile(ticks_per_beat=480)
                
                # 시작, 종료 틱 계산
                start_tick = split_index * ticks_per_bar * bars
                end_tick = min((split_index + 1) * ticks_per_bar * bars, total_ticks)
                
                # 메타 트랙 생성
                meta_track = MidiTrack()
                new_midi.tracks.append(meta_track)
                
                # 메타 이벤트 복사 (템포, 박자표)
                for track in midi.tracks:
                    for msg in track:
                        if msg.is_meta and (msg.type == 'set_tempo' or msg.type == 'time_signature'):
                            meta_track.append(msg.copy())
                
                # 내용 트랙 생성
                for track in midi.tracks[1:]:
                    content_track = MidiTrack()
                    new_midi.tracks.append(content_track)
                    current_tick = 0
                    # 트랙 이름, 악기 종류 메시지 복사
                    for msg in track:
                        if split_index != 0:
                            if (msg.is_meta and msg.type == 'track_name') or msg.type == 'program_change':
                                content_track.append(msg.copy())

                        current_tick += msg.time

                        if start_tick <= current_tick <= end_tick:
                            if current_tick - msg.time < start_tick:
                                new_msg = msg.copy()
                                new_msg.time = current_tick - start_tick
                                content_track.append(new_msg)
                            else:
                                content_track.append(msg.copy())
                
                # 마지막 분할에서 패딩
                if split_index == split_count - 1 and end_tick < bars * ticks_per_bar * split_count:
                    padding_ticks = bars * ticks_per_bar * split_count - end_tick
                    # 패딩 메시지 추가
                    content_track.append(Message('note_on', note=0, velocity=0, time=padding_ticks))

                result.append(new_midi)
        return result
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        return None

'''
8마디씩 분할한 것들을 한 마디씩 분할
파일 읽기 필요 없음
파일로 저장하지 않고 리턴
'''
def split_midi_by_bar(tracks):
    try:
        '''
        ticks_per_bar = 480 * 4 = 1920
        마디 당 틱 수 = 비트 당 틱 수 * 마디 당 비트 수
        대부분의 DAW에서 사용하는 ticks per beat는 480
        학습에 사용할 데이터는 4/4 박자
        '''
        ticks_per_bar = 1920
        
        track_lengths = []

        for track in tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
            track_lengths.append(track_time)

        if track_lengths:
            max_index, total_ticks = max(enumerate(track_lengths), key=lambda x: x[1])
        else:
            max_index, total_ticks = None, 0
        
        # 결과
        result = []

        # 8마디로 분할한 것들을 한 마디씩 분할
        split_count = 8
        
        # 각 부분별로 처리
        for split_index in range(split_count):
            new_midi = MidiFile(ticks_per_beat=480)
            
            # 시작/종료 틱 계산
            start_tick = split_index * ticks_per_bar
            end_tick = min((split_index + 1) * ticks_per_bar, total_ticks)
            
            # 메타 트랙 생성
            meta_track = MidiTrack()
            new_midi.tracks.append(meta_track)

            # 메타 이벤트 복사 (템포, 박자표)
            for track in midi.tracks:
                for msg in track:
                    if msg.is_meta and (msg.type == 'set_tempo' or msg.type == 'time_signature'):
                        meta_track.append(msg.copy())

            # 내용 트랙 생성
            for track in midi.tracks[1:]:
                content_track = MidiTrack()
                new_midi.tracks.append(content_track)
                current_tick = 0
                # 트랙 이름, 악기 종류 메시지 복사
                for msg in track:
                    if split_index != 0:
                        if (msg.is_meta and msg.type == 'track_name') or msg.type == 'program_change':
                            content_track.append(msg.copy())
                    current_tick += msg.time

                    if start_tick <= current_tick <= end_tick:
                        if current_tick - msg.time < start_tick:
                            new_msg = msg.copy()
                            new_msg.time = current_tick - start_tick
                            content_track.append(new_msg)
                        else:
                            content_track.append(msg.copy())
            
            # 마지막 분할에서 패딩
            if split_index == split_count - 1 and end_tick < ticks_per_bar * split_count:
                padding_ticks = ticks_per_bar * split_count - end_tick
                # 패딩 메시지 추가
                content_track.append(Message('note_on', note=0, velocity=0, time=padding_ticks))
            result.append(new_midi)

        return result
    
    except Exception as e:
        print(f"에러 발생: {str(e)}")
    finally:
        return result

'''
베이스 트랙, 드럼 트랙, 다른 악기들 트랙으로 분리
분리하면서 0번 트랙 삭제
0번 트랙은 BPM 정보와 박자표 정보만 있고 연주 정보는 없음
'''
def extract_tracks(midi):
    bass_tracks = []
    drum_tracks = []
    other_tracks = []
    
    for track in midi.tracks:
        track_name = ""
        has_drum_channel = False
        has_bass_hint = False

        # 트랙 이름 및 채널 확인
        for msg in track:
            if msg.is_meta and msg.type == 'track_name':
                track_name = msg.name.lower()
            elif not msg.is_meta and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    has_drum_channel = True
                if msg.type == 'program_change' and msg.program in range(32, 40):  # 베이스 계열
                    has_bass_hint = True
        
        # 트랙 분류
        if has_drum_channel or 'drum' in track_name:
            drum_tracks.append(track)
        elif 'bass' in track_name or has_bass_hint:
            bass_tracks.append(track)
        else:
            other_tracks.append(track)

    return bass_tracks, drum_tracks, other_tracks[1:]

'''
각 트랙에서 시작하는 메타데이터 삭제
이미 악기별로 트랙을 분할했기 때문에 트랙 이름과 악기 종류를 나타내는 정보는 필요가 없어짐
'''
def remove_meta_data(tracks):
    for track in tracks:
        if len(track) >= 2:
            if track[0].is_meta and track[1].type == 'program_change':
                del track[0:2]
        if track:
            if track[-1].is_meta and track[-1].type == 'end_of_track':
                del track[-1]

def save_midi_messages(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_midi_messages(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

'''
MidiFile(type=1, ticks_per_beat=480, tracks=[
    MidiTrack([
        Message('note_on', channel=1, note=40, velocity=0, time=240),
        Message('note_on', channel=1, note=43, velocity=80, time=0),
        Message('note_on', channel=1, note=47, velocity=0, time=0),
        Message('note_on', channel=1, note=50, velocity=80, time=0),
        Message('note_on', channel=1, note=52, velocity=0, time=0)
    ])
])
MIDI의 메시지 리스트를 이벤트 리스트로 변환
event {
    'type': event_type,
    'time': absolute_time,
    'note': note,
    'velocity': velocity,
    'channel': channel
}
'''
def convert_to_events(track):
    events = []
    absolute_time = 0

    for msg in track:
        absolute_time += msg.time

        if msg.type == 'note_on':
            event_type = 'note_on' if msg.velocity > 0 else 'note_off'
            events.append({
                'type': event_type,
                'time': absolute_time,
                'note': msg.note,
                'velocity': msg.velocity,
                'channel': msg.channel
            })
    events.sort(key=lambda x: x['time'])
    return events

'''
변환된 이벤트 리스트 중 같은 시간에 일어난 것들을 그룹화
grouped_event {
    (
        time1, [
            event1,
            event2,
            ...
        ]
    ),
    (
        time2, [
            event1,
            event2,
            ...
    ]),
    ...
}
'''
def convert_to_grouped_events(events):
    if not events:
        return []
    
    grouped_events = []
    current_time = events[0]['time']
    current_group = []
    
    for event in events:
        if event['time'] == current_time:
            current_group.append(event)
        else:
            grouped_events.append((current_time, current_group))
            current_time = event['time']
            current_group = [event]
    
    # 마지막 그룹 추가
    if current_group:
        grouped_events.append((current_time, current_group))
    
    return grouped_events

'''
그룹화된 이벤트를 remi_events로 변환
remi {[
    {
        'type': 'Position',
        'value': int { 0 - 15 }
    },
    {
        'type': 'Pitch',
        'value': int { 0 - 127 }
    },
    {
        'type': 'Velocity',
        'value': int { 0 - 127 }
    },
    {
        'type': 'Duration',
        'value': int // ticks
    },
    ...
]}
'''
def convert_to_remi_events(grouped_events):

    remi_events = []
    current_bar = 0
    current_position = 0
    bar_length = 1920

    '''
    active_notes = {
        (channel, note): absolute_time
    }
    '''
    active_notes = {}

    for time, events in grouped_events:
        position_in_bar = time % bar_length

        if position_in_bar != current_position:
            current_position = position_in_bar
            position_index = int((position_in_bar / bar_length) * 16)
            remi_events.append({'type': 'Position', 'value': position_index})

        for event in events:
            note_id = (event['channel'], event['note'])

            if event['type'] == 'note_on':
                if event['channel'] == 9:
                    remi_events.append({'type': 'Drumhit', 'value': True})
                else:
                    remi_events.append({'type': 'Pitch', 'value': event['note']})

                remi_events.append({'type': 'Velocity', 'value': event['velocity']})
                active_notes[note_id] = time

            elif event['type'] == 'note_off':
                if note_id in active_notes:
                    start_time = active_notes[note_id]
                    duration_ticks = time - start_time

                    remi_events.append({'type': 'Duration', 'value': duration_ticks})
                    del active_notes[note_id]
                # 마디 분할 때문에 마디 내 note_on이 없는 것들 처리
                else:
                    remi_events.append({'type': 'Duration', 'value': time})
    # 마디 분할 때문에 마디 내 note_off가 없는 것들 처리
    if active_notes:
        for note in active_notes:
            remi_events.append({'type': 'Duration', 'value': bar_length - active_notes[note_id]})
    return remi_events

'''
remi에서 duration을 틱으로 표현하지 않고 양자화함
'''
def convert_to_quantize_remi_events(remi_events):
    '''
    대부분의 DAW에서 사용하는 ticks per beat는 480
    '''
    ticks_per_beat = 480

    durations = [
        ticks_per_beat * 4,      # 온음표
        ticks_per_beat * 2,      # 2분음표
        ticks_per_beat,          # 4분음표
        ticks_per_beat // 2,     # 8분음표 
        ticks_per_beat // 4,     # 16분음표
        ticks_per_beat // 8,     # 32분음표
        ticks_per_beat * 3 // 2, # 점4분음표
        ticks_per_beat * 3 // 4, # 점8분음표
        ticks_per_beat * 3 // 8  # 점16분음표
    ]
    
    quantized_events = []

    for event in remi_events:
        if event['type'] == 'Duration':
            duration_ticks = event['value']
            closest_duration = min(durations, key=lambda d: abs(d - duration_ticks))
            duration_index = durations.index(closest_duration)
            quantized_events.append({'type': 'Duration', 'value': duration_index})
        else:
            quantized_events.append(event)
    
    return quantized_events

def convert_to_token(remi_events):
    token_map = {
        'Bar': 0,
        'Position': list(range(1, 17)),
        'Pitch': list(range(17, 17+128)),
        'Duration': list(range(145, 145+9)),
        'Velocity': list(range(154, 154+32))
    }

    tokens = []
    for event in remi_events:
        if event['type'] == 'Position':
            position_token = token_map['Position'][event['value']]
            tokens.append(position_token)
        elif event['type'] == 'Pitch':
            pitch_token = token_map['Pitch'][event['value']]
            tokens.append(pitch_token)
        elif event['type'] == 'Duration':
            duration_token = token_map['Duration'][event['value']]
            tokens.append(duration_token)
        elif event['type'] == 'Velocity':
            # 0-127 벨로시티를 0-31 범위로 양자화
            velocity_level = min(31, event['value'] // 4)
            velocity_token = token_map['Velocity'][velocity_level]
            tokens.append(velocity_token)
    
    return tokens


if __name__ == '__main__':

    input_dir = os.path.join('midi','test')
    output_dir = 'pkl'

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.midi'):
            midi_path = os.path.join(input_dir, file_name)
            midi_name = os.path.splitext(file_name)[0] + '.midi'
            output_path = os.path.join(output_dir, midi_name)

            midis = split_midi_by_bars(os.path.join(input_dir, file_name))

            for i, midi in enumerate(midis):
                midi_splited_by_bar = split_midi_by_bar(midi.tracks)

                for ii, tracks in enumerate(midi_splited_by_bar):
                    bass_tracks, drum_tracks, other_tracks = extract_tracks(tracks)
                    remove_meta_data(bass_tracks)
                    remove_meta_data(drum_tracks)
                    remove_meta_data(other_tracks)

                    for iii, bass in enumerate(bass_tracks):
                        path = os.path.join(output_dir, f'{file_name}_bass_{i}_{ii}_{iii}.pkl')
                        save_midi_messages(bass, path)
                    for iii, drum in enumerate(drum_tracks):
                        path = os.path.join(output_dir, f'{file_name}_drum_{i}_{ii}_{iii}.pkl')
                        save_midi_messages(drum, path)
                    for iii, other in enumerate(other_tracks):
                        path = os.path.join(output_dir, f'{file_name}_other_{i}_{ii}_{iii}.pkl')
                        save_midi_messages(other, path)

    print('done')
