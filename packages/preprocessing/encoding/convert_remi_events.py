from . import token_map

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
def convert_to_quantize_remi_events(remi_events, durations=token_map.DURATION):
    '''
    대부분의 DAW에서 사용하는 ticks per beat는 480
    '''
    ticks_per_beat = 480
    
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
