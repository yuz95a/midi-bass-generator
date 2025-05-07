import os
import pickle
import statistics
from mido import MidiFile, MidiTrack, Message, merge_tracks

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

    _remi_events = []
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
        position_index = int((position_in_bar / bar_length) * 16) # position quantize
        
        # time이 1920인 것이 있다면 다음 마디의 0과 겹침
        if time == 1920:
            break

        '''
        time과 이벤트들을 묶어서 remi로 변환
        duration 이벤트는 time이 아니라 start time과 매핑
        position 이벤트는 있지만 그 외에는 아무 이벤트도 없는 문제 발생
        '''
        grouped_remi = []
        grouped_remi.append({'type': 'Position', 'value': position_index})

        for event in events:
            note_id = (event['channel'], event['note'])

            if event['type'] == 'note_on':
                if event['channel'] == 9:
                    grouped_remi.append({'type': 'Drumhit', 'value': True})
                else:
                    grouped_remi.append({'type': 'Pitch', 'value': event['note']})

                grouped_remi.append({'type': 'Velocity', 'value': event['velocity']})
                active_notes[note_id] = time

            elif event['type'] == 'note_off':
                if note_id in active_notes:
                    start_time = active_notes[note_id]
                    duration_ticks = time - start_time

                    _remi_events.append((start_time, [{'type': 'Duration', 'value': duration_ticks}]))
                    del active_notes[note_id]
                # 마디 분할 때문에 마디 내 note_on이 없는 것들 처리
                else:
                    _remi_events.append((0, [{'type': 'Duration', 'value': time}]))
        
        _remi_events.append((time, grouped_remi))

    # 마디 분할 때문에 마디 내 note_off가 없는 것들 처리
    if active_notes:
        for note_id in active_notes:
            _remi_events.append((active_notes[note_id], [{'type': 'Duration', 'value': bar_length - active_notes[note_id]}]))

    # 시간 별로 정렬 후 이벤트만 추출
    _remi_events.sort(key=lambda x: x[0])

    '''
    Position 이벤트만 있고 Pitch나 Drumhit 이벤트가 없으면 제외
    '''
    remi_events = []
    for events in _remi_events:
        for remi in events[1]:
            if remi['type'] == 'Position' and len(events[1]) == 1:
                continue
            remi_events.append(remi)

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

def convert_to_structured_remi_events(remi_events):
    position0 = False
    play = 0
    duration = 0
    structured_events = []
    for event in remi_events:
        # Position 0 이전의 이벤트 삭제
        if event['type'] == 'Position':
            position0 = True
        if not position0 and event['type'] != 'Position':
            continue

        # Pitch or Drumhit 이벤트, Velocity 이벤트, Duration 이벤트 묶기
        if event['type'] == 'Pitch' or event['type'] == 'Drumhit':
            play += 1
            structured_events.append((play, event))
        elif event['type'] == 'Velocity':
            structured_events.append((play, event))
        elif event['type'] == 'Duration':
            duration += 1
            structured_events.append((duration, event))
        else:
            structured_events.append((play, event))
        
    structured_events.sort(key=lambda x: x[0])

    position = 0
    structured_remi_events = []
    for event in structured_events:
        if event[1]['type'] == 'Position':
            position = event[1]['value']
        structured_remi_events.append((position, event[1]))

    return structured_remi_events

def convert_to_token(remi_events, eos=False):
    token_map = {
        'PAD': 0, # 패딩용 토큰
        'Position': list(range(1, 1 + 16)),
        'Drumhit': 17,
        'Pitch': list(range(18, 18 + 128)),
        'Velocity': list(range(146, 146 + 32)),
        'Duration': list(range(178, 178 + 9)),
        'MASK': 187, # 마스킹
        'BOS': 188, # 시작
        'EOS': 189, # 끝
        'BAR': 190, # 마디 구분
        'DUMMY': 191 # vocab size를 8의 배수로 맞추기 위한 더미
    }

    tokens = []

    if not remi_events:
        return tokens
    
    tokens.append(token_map['BOS'])
    for event in remi_events:
        if event['type'] == 'Position':
            position_token = token_map['Position'][event['value']]
            tokens.append(position_token)
        elif event['type'] == 'Drumhit':
            drum_token = token_map['Drumhit']
            tokens.append(drum_token)
        elif event['type'] == 'Pitch':
            pitch_token = token_map['Pitch'][event['value']]
            tokens.append(pitch_token)
        elif event['type'] == 'Velocity':
            # 0-127 벨로시티를 0-31 범위로 양자화
            velocity_level = min(31, event['value'] // 4)
            velocity_token = token_map['Velocity'][velocity_level]
            tokens.append(velocity_token)
        elif event['type'] == 'Duration':
            duration_token = token_map['Duration'][event['value']]
            tokens.append(duration_token)
    
    tokens.append(token_map['BAR'])

    if eos:
        tokens.append(token_map['EOS'])

    return tokens

def combine_remi_events(*remi_events):
    result_events = []
    if len(remi_events) == 1:
        result_events = [event[1] for event in remi_events[0]]
    else:
        for structured_events in remi_events:
            for events in structured_events:
                result_events.append(events)
        result_events.sort(key=lambda x: x[0])
        result_events = [event[1] for event in result_events]

    # Pitch 이벤트, Drumhit 이벤트 앞에 Position 이벤트가 누락되는 현상 방지
    position = 0
    i = 0
    while i < len(result_events):
        if result_events[i]['type'] == 'Position':
            position = result_events[i]['value']
        elif result_events[i]['type'] == 'Pitch' or result_events[i]['type'] == 'Drumhit':
            if result_events[i - 1]['type'] != 'Position':
                result_events.insert(i, {'type': 'Position', 'value': position})
                i += 1
        i += 1
    return result_events


def print_statistics(data, title=None):
    if not data:
        print("리스트가 비어 있습니다.")
        return

    max_val = max(data)
    min_val = min(data)
    avg_val = sum(data) / len(data)
    median_val = statistics.median(data)
    std_dev = statistics.stdev(data) if len(data) > 1 else 0.0

    if title:
        print(f"{'-'*40}{title}{'-'*40}")
    print(f"최댓값: {max_val}")
    print(f"최솟값: {min_val}")
    print(f"평균: {avg_val:.2f}")
    print(f"중앙값: {median_val}")
    print(f"표준편차: {std_dev:.2f}")


if __name__ == '__main__':

    input_dir = os.path.join('midi','samble')
    output_dir = os.path.join('token','test')

    feature_len = []
    label_len = []

    for file_name in os.listdir(input_dir): # 각 파일에 대해서
        if file_name.endswith('.midi'):
            midi_path = os.path.join(input_dir, file_name)
            midi_name = os.path.splitext(file_name)[0] + '.midi'
            output_path = os.path.join(output_dir, midi_name)

            midis = split_midi_by_bars(os.path.join(input_dir, file_name))

            for i, midi in enumerate(midis): # 각 midi 파일의 8마디에 대해서
                midi_splited_by_bar = split_midi_by_bar(midi.tracks)

                for ii, tracks in enumerate(midi_splited_by_bar): # 각 마디에 대해서
                    bass_tracks, drum_tracks, other_tracks = extract_tracks(tracks)
                    remove_meta_data(bass_tracks)
                    remove_meta_data(drum_tracks)
                    remove_meta_data(other_tracks)

                    bass_events = []
                    for iii, bass in enumerate(bass_tracks): # 베이스 트랙 한 마디에 대해서
                        path = os.path.join(output_dir, f'{file_name}_bass_{i}_{ii}_{iii}.pkl')

                        events = convert_to_events(bass)
                        grouped_events = convert_to_grouped_events(events)
                        remi_events = convert_to_remi_events(grouped_events)
                        quantized_events = convert_to_quantize_remi_events(remi_events)
                        structured_events = convert_to_structured_remi_events(quantized_events)
                        # token = convert_to_token(quantized_events)
                        bass_events.append(structured_events)

                    drum_events = []
                    for iii, drum in enumerate(drum_tracks): # 드럼 트랙 한 마디에 대해서
                        path = os.path.join(output_dir, f'{file_name}_drum_{i}_{ii}_{iii}.pkl')

                        events = convert_to_events(drum)
                        grouped_events = convert_to_grouped_events(events)
                        remi_events = convert_to_remi_events(grouped_events)
                        quantized_events = convert_to_quantize_remi_events(remi_events)
                        structured_events = convert_to_structured_remi_events(quantized_events)
                        # token = convert_to_token(quantized_events)
                        drum_events.append(structured_events)

                    other_events = []
                    for iii, other in enumerate(other_tracks): # 다른 악기 트랙 한 마디에 대해서
                        path = os.path.join(output_dir, f'{file_name}_other_{i}_{ii}_{iii}.pkl')

                        events = convert_to_events(other)
                        grouped_events = convert_to_grouped_events(events)
                        remi_events = convert_to_remi_events(grouped_events)
                        quantized_events = convert_to_quantize_remi_events(remi_events)
                        structured_events = convert_to_structured_remi_events(quantized_events)
                        # token = convert_to_token(quantized_events)
                        other_events.append(structured_events)
                

                feature_events = combine_remi_events(*drum_events, *other_events)
                feature_tokens = convert_to_token(feature_events)

                label_events = combine_remi_events(*bass_events)
                label_tokens = convert_to_token(label_events)

                if feature_tokens and label_tokens:
                    # 트랙의 마지막 부분에 EOS 토큰 삽입
                    if i == len(midis) - 1 and ii == len(midi_splited_by_bar) - 1:
                        feature_tokens.append(2) # token_map['EOS'] = 2
                        label_tokens.append(2) # token_map['EOS'] = 2

                    path = os.path.join(output_dir, f'{file_name}_token_{i}_{ii}.pkl')
                    data = {'feature': feature_tokens, 'label': label_tokens}
                    save_midi_messages(data, path)
                    print(f'{file_name}_token_{i}_{ii}.pkl 저장 완료')

                    feature_len.append(len(feature_tokens))
                    label_len.append(len(label_tokens))
              
    print_statistics(feature_len, "feature")
    print_statistics(label_len, "label")
