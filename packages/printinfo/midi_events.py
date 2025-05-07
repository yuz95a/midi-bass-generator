from mido import MidiFile

def print_midi_events(filename):
    mid = MidiFile(filename)

    print(f"파일: {filename}")
    print(f"ticks_per_beat: {mid.ticks_per_beat}")
    print("-" * 80)

    for i, track in enumerate(mid.tracks):
        print(f"\n--- 트랙 {i}: {track.name} ---")
        for msg in track:
            if msg.is_meta:
                args = ", ".join(f"{key}={repr(value)}" for key, value in msg.dict().items() if key != 'type')
                print(f"MetaMessage('{msg.type}', {args})")
            else:
                args = ", ".join(f"{key}={repr(value)}" for key, value in msg.dict().items() if key != 'type')
                print(f"Message('{msg.type}', {args})")