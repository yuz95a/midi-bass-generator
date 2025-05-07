from mido import MidiFile, Message
import os

'''
파일 마지막 노트 이벤트 출력
'''
def print_note_off(midi_path):
    midi_file = MidiFile(midi_path)

    for i, track in enumerate(midi_file.tracks):
        if len(track) < 2:
            print(f'트랙 {i}: 이벤트 수가 부족합니다 (길이 {len(track)}).')
            continue

        second_last_event = track[-1]

        if isinstance(second_last_event, Message) and second_last_event.type == 'note_off':
            print(f'트랙 {i}: ✅ 마지막에서 두 번째 메시지는 note_off입니다.')
        else:
            print(f'트랙 {i}: ❌ 마지막에서 두 번째 메시지는 note_off가 아닙니다. (type: {second_last_event.type})')
'''
디렉토리 내 모든 midi 파일의 마지막 노트 이벤트 출력
'''
def check_note_off_directory(directory_path):
    for i, filename in enumerate(sorted(os.listdir(directory_path))):
        if not filename.endswith('.midi'):
            continue

        else:
            print(f'{filename}을 검사합니다.')
            midi_path = os.path.join(directory_path, filename)
            try:
                print_note_off(midi_path)
            except Exception as e:
                print(f'Error processing {filename}: {e}')
