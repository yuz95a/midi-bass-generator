from mido import MidiFile, MetaMessage
import os

'''
파일 end of track 이벤트 출력
'''
def print_end_of_track(path):
    mid = MidiFile(path)

    for i, track in enumerate(mid.tracks):
        if not track:
            print(f"트랙 {i}는 비어 있습니다.")
            continue

        last_event = track[-1]

        if isinstance(last_event, MetaMessage) and last_event.type == 'end_of_track':
            print(f"트랙 {i}: ✅ 마지막 이벤트는 end_of_track입니다.")
        else:
            print(f"트랙 {i}: ❌ 마지막 이벤트는 end_of_track이 아닙니다. (type: {last_event.type})")
'''
디렉토리 내 모든 midi 파일의 end of track 이벤트 출력
'''
def check_end_of_track_directory(directory_path):
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if not filename.endswith(".midi"):
            continue

        if filename.endswith(".midi"):
            print(f"{filename}을 검사합니다.")
            midi_path = os.path.join(directory_path, filename)
            try:
                print_end_of_track(midi_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
