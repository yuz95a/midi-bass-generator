from mido import MetaMessage, Message, MidiFile, MidiTrack, bpm2tempo

def convert_to_midi(midi_events, inst=32, bpm=120, numerator=4, denominator=4):
    midi = MidiFile(ticks_per_beat=480)

    meta_track = MidiTrack()
    midi.tracks.append(meta_track)

    content_track = MidiTrack()
    midi.tracks.append(content_track)
    
    tempo = bpm2tempo(bpm) # 설정: 120 BPM, 4/4 박자
    meta_track.append(MetaMessage('set_tempo', tempo=tempo))
    meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4))

    content_track.append(MetaMessage('track_name', name='bass'))
    content_track.append(Message('program_change', program=inst))

    for e in midi_events:
        if e['type'] == 'note_on':
            content_track.append(Message('note_on', note=int(e['pitch']), velocity=int(e['velocity']), time=int(e['delta'])))
        elif e['type'] == 'note_off':
            content_track.append(Message('note_on', note=int(e['pitch']), velocity=0, time=int(e['delta'])))

    return midi