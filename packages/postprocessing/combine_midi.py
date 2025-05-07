from mido import MetaMessage, Message, MidiFile, MidiTrack, bpm2tempo

def combine_midis(midis, inst=32, bpm=120):
    new_midi = MidiFile(ticks_per_beat=480)

    meta_track = MidiTrack()
    new_midi.tracks.append(meta_track)

    content_track = MidiTrack()
    new_midi.tracks.append(content_track)

    tempo = bpm2tempo(bpm) # 설정: 120 BPM, 4/4 박자
    meta_track.append(MetaMessage('set_tempo', tempo=tempo))
    meta_track.append(MetaMessage('time_signature', numerator=4, denominator=4))

    content_track.append(MetaMessage('track_name', name='bass'))
    content_track.append(Message('program_change', program=inst))

    for midi in midis:
        for track in midi.tracks[1:]:
            for msg in track:
                content_track.append(msg.copy())

    return new_midi