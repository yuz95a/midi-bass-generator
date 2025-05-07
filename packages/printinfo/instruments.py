from mido import MidiFile

def print_instruments(filename):
    midi = MidiFile(filename)

    print(f"파일: {filename}")
    print(f"ticks_per_beat: {midi.ticks_per_beat}")
    print("-" * 80)
    
    for i, track in enumerate(midi.tracks):
        print(f"\n--- 트랙 {i}: {track.name} ---")
        for msg in track:
            if msg.is_meta and msg.type == 'track_name':
                print(msg.name)
                break

def print_instruments_directory(input_dir):
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.midi'):
            mid = MidiFile(os.path.join(input_dir, file_name))
            if len(mid.tracks) > 4:
                print('-'*80)
                print(f'{file_name}')
                for i, track in enumerate(mid.tracks):
                    print(f'\n{"-"*30} 트랙 {i}: {track.name} {"-"*30}')
                    for msg in track:
                        if msg.is_meta and msg.type == 'track_name':
                            print(msg.name)
                            break