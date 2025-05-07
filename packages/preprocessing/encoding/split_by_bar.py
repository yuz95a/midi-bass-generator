from mido import MidiFile, MidiTrack

def split_midi_by_bar(midi_path):
    try:
        midi = MidiFile(midi_path)
        midis = []
        '''
        ticks_per_bar = 480 * 4 = 1920
        마디 당 틱 수 = 비트 당 틱 수 * 마디 당 비트 수
        대부분의 DAW에서 사용하는 ticks per beat는 480
        학습에 사용할 데이터는 4/4 박자
        '''
        ticks_per_bar = 1920
        
        track_lengths = []

        for track in midi.tracks:
            track_time = 0
            for msg in track:
                track_time += msg.time
            track_lengths.append(track_time)
        
        total_ticks = max(track_lengths)

        split_count = (total_ticks + ticks_per_bar - 1) // ticks_per_bar
        
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

            midis.append(new_midi)
        return midis
    
    except Exception as e:
        print(f'ERROR: {str(e)}')
        return None
