import pretty_midi
import sys
import numpy as np
from PIL import Image
import torch
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
                # MetaMessage 예시 출력
                args = ", ".join(f"{key}={repr(value)}" for key, value in msg.dict().items() if key != 'type')
                print(f"MetaMessage('{msg.type}', {args})")
            else:
                # 일반 Message 예시 출력
                args = ", ".join(f"{key}={repr(value)}" for key, value in msg.dict().items() if key != 'type')
                print(f"Message('{msg.type}', {args})")

def print_instruments(filename):
    mid = MidiFile(filename)

    print(f"파일: {filename}")
    print(f"ticks_per_beat: {mid.ticks_per_beat}")
    print("-" * 80)
    
    for i, track in enumerate(mid.tracks):
        print(f"\n--- 트랙 {i}: {track.name} ---")
        for msg in track:
            if msg.is_meta and msg.type == 'track_name':
                print(msg.name)
                break

def print_midi_events_old(file_path):
    # MIDI 파일 로드
    midi_data = pretty_midi.PrettyMIDI(file_path)

    print(f"\n🎼 MIDI 파일: {file_path}")
    print(f"⏱ 템포: {midi_data.get_tempo_changes()[1]} BPM (시점: {midi_data.get_tempo_changes()[0]})")
    print(f"🎼 키 시그니처:")
    for key in midi_data.key_signature_changes:
        print(f"  - {key.time:.2f}s: {key.key}")

    print("\n🎹 트랙 정보:")
    for i, instrument in enumerate(midi_data.instruments):
        print(f"\n[{i}] 프로그램 번호: {instrument.program} ({'드럼' if instrument.is_drum else '악기'})")
        print(f"노트 개수: {len(instrument.notes)}")
        for note in instrument.notes:
            print(f"  - 시간: {note.start:.2f}s ~ {note.end:.2f}s, 음: {pretty_midi.note_number_to_name(note.pitch)}, 속도: {note.velocity}")

    print("\n🎛 컨트롤 체인지 이벤트:")
    for i, instrument in enumerate(midi_data.instruments):
        for cc in instrument.control_changes:
            print(f"  - 트랙 {i}, 시간: {cc.time:.2f}s, 컨트롤 번호: {cc.number}, 값: {cc.value}")



def save_tensor_as_image(tensor, filename="output.png"):
    """
    2차원 텐서를 이미지로 저장합니다.

    Parameters:
        tensor (torch.Tensor): 2차원 텐서 (H x W)
        filename (str): 저장할 이미지 파일 이름 (기본값: output.png)
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("입력은 torch.Tensor여야 합니다.")
    
    if tensor.ndim != 2:
        raise ValueError("입력 텐서는 2차원이어야 합니다. (H x W)")

    # 텐서를 NumPy 배열로 변환
    array = tensor.cpu().numpy()

    # 값을 0~255 범위로 정규화
    array = (array * 255).astype(np.uint8)

    # NumPy 배열을 PIL 이미지로 변환
    image = Image.fromarray(array)

    # 이미지 저장
    image.save(filename)
    print(f"이미지를 {filename}로 저장했습니다.")


def duration_bins(resolution=24):
        max_duration = 8 * resolution  # 최대 2마디 길이
        duration_bins = [0]
        current_tick = 1
        while current_tick < max_duration:
            duration_bins.append(current_tick)
            if current_tick < 16:
                current_tick += 1
            elif current_tick < 32:
                current_tick += 2
            elif current_tick < 64:
                current_tick += 4
            elif current_tick < 128:
                current_tick += 8
            else:
                current_tick += 16
        
        for i in duration_bins:
            print(i, end=' ')

def position_bins(resolution=24):
    position_bins = np.arange(0, resolution, 1)

    for i in position_bins:
            print(i, end=' ')

def tempo_bins(resolution=24):
    tempo_bins = np.arange(30, 210, 5)
    for i in tempo_bins:
                print(i, end=' ')

if __name__ == "__main__":
    # if len(sys.argv) == 2:
    #     print("사용법: python midi_events.py <midi파일경로>")
    # elif len(sys.argv) == 1:
    #     print_midi_events(sys.argv[1])
    # else:
    # tempo_bins()
    print_instruments(sys.argv[1])
    # print_midi_events(sys.argv[1])
