from mido import Message, MidiFile, MidiTrack, MetaMessage, bpm2tempo

def create_8bar_midi(filename="8bar_scale_test.mid"):
    mid = MidiFile()

    tracks = []
    programs = [53, 2, 10, 34, 12, 26, 13, 68]
    '''
    53: Voice Oohs
     2: Electric Grand Piano
    10: Music Box
    34: Electric Bass (finger)
    12: Marimba
    26: Electric Guitar (jazz)
    13: Xylophone
    68: Oboe
    '''

    for i in range(len(programs) + 1):
        track = MidiTrack()
        mid.tracks.append(track)
        tracks.append(track)

    append_instrument(tracks[0])

    for i, inst in enumerate(programs):
        append_instrument(track=tracks[i+1], channel=i+1, inst=inst)
        play_track(mid=mid, track=tracks[i+1], channel=i+1)

    mid.save(filename)
    print(f"MIDI 파일 저장됨: {filename}")

def append_instrument(track, channel=0, inst=0, tempo=None):
    if channel == 0:
        tempo = bpm2tempo(120) # 설정: 120 BPM, 4/4 박자
        track.append(MetaMessage('set_tempo', tempo=tempo))
        track.append(MetaMessage('time_signature', numerator=4, denominator=4))
    else:
        track.append(Message('program_change', program=inst, channel=channel))

def play_track(mid, track, channel):
    ticks_per_beat = mid.ticks_per_beat  # 기본 480
    quarter_note = ticks_per_beat        # 4분음표는 1 비트

    cs = [48, 60]#, 24, 36] # C3, C4, C1, C2

    for i in range(len(cs)):
        for ii in range(13):
            if ii in [1, 3, 6, 8, 10]:
                continue

            track.append(Message('note_on', note=cs[i]+ii, velocity=64, time=0, channel=channel))
            track.append(Message('note_on', note=cs[i]+ii, velocity=0, time=quarter_note, channel=channel))
            # track.append(Message('note_on', note=cs[i]+ii, velocity=64, time=0, channel=channel))
            # track.append(Message('note_off', note=cs[i]+ii, velocity=64, time=quarter_note, channel=channel))
    track.pop()

# 실행
create_8bar_midi("8bar_scale_test19_on.midi")
