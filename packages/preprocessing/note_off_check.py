from mido import MidiFile, Message
import os

def check_note_off(path):
    mid = MidiFile(path)

    for i, track in enumerate(mid.tracks):
        if len(track) < 2:
            print(f"트랙 {i}: 이벤트 수가 부족합니다 (길이 {len(track)}).")
            continue

        second_last_event = track[-1]

        if isinstance(second_last_event, Message) and second_last_event.type == 'note_off':
            print(f"트랙 {i}: ✅ 마지막에서 두 번째 메시지는 note_off입니다.")
        else:
            print(f"트랙 {i}: ❌ 마지막에서 두 번째 메시지는 note_off가 아닙니다. (type: {second_last_event.type})")

def check_note_off_directory(directory_path):
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if not filename.endswith(".midi"):
            continue

        if filename.endswith(".midi"):
            print(f"{filename}을 검사합니다.")
            midi_path = os.path.join(directory_path, filename)
            try:
                check_note_off(midi_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    check_note_off_directory(os.path.join('midi', 'test'))