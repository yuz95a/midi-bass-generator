import os
from mido import MidiFile, MidiTrack, MetaMessage, bpm2tempo

def change_tempo_to_120(input_path, output_path):
    """
    단일 MIDI 파일을 불러와 템포를 120BPM으로 설정하여 저장합니다.
    """
    midi = MidiFile(input_path)
    new_midi = MidiFile()
    new_midi.ticks_per_beat = midi.ticks_per_beat

    tempo_meta = MetaMessage('set_tempo', tempo=bpm2tempo(120))

    for i, track in enumerate(midi.tracks):
        new_track = MidiTrack()
        new_midi.tracks.append(new_track)

        for msg in track:
            if msg.type == 'set_tempo':
                continue
            new_track.append(msg)

        if i == 0:
            new_track.insert(0, tempo_meta)

    new_midi.save(output_path)

def convert_directory_to_120bpm(input_dir, output_dir):
    """
    디렉토리 내 모든 MIDI 파일을 120BPM으로 바꿔서 output_dir에 저장합니다.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.mid', '.midi')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            try:
                change_tempo_to_120(input_path, output_path)
                print(f"변환 완료: {filename}")
            except Exception as e:
                print(f"변환 실패: {filename} - {e}")

# 사용 예시
input_directory = os.path.join('midi', 'test')
output_directory = "converted_120bpm_midis"  # 결과 저장할 디렉토리

convert_directory_to_120bpm(input_directory, output_directory)
