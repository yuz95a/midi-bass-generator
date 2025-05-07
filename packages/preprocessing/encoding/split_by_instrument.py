def split_midi_by_instrument(midi):
    bass_tracks = []
    drum_tracks = []
    other_tracks = []
    
    for track in midi.tracks:
        track_name = ""
        has_drum_channel = False
        has_bass_hint = False

        for msg in track:
            if msg.is_meta and msg.type == 'track_name':
                track_name = msg.name.lower()
            elif not msg.is_meta and hasattr(msg, 'channel'):
                if msg.channel == 9:
                    has_drum_channel = True
                if msg.type == 'program_change' and msg.program in range(32, 40):
                    has_bass_hint = True

        if has_drum_channel or 'drum' in track_name:
            drum_tracks.append(track)
        elif 'bass' in track_name or has_bass_hint:
            bass_tracks.append(track)
        else:
            other_tracks.append(track)

    return bass_tracks, drum_tracks, other_tracks[1:]