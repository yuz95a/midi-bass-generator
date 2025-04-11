import torch
import pretty_midi
import transformer
import numpy as np
from safetensors.torch import load_file
import os
import argparse

def load_model(model_path, device):
    """모델 로드"""
    model = transformer.MIDIBassGenerator(
        input_dim=128,
        hidden_dim=256,
        num_layers=3,
        output_dim=128
    ).to(device)
    
    # safetensors 파일에서 state_dict 로드
    state_dict = load_file(model_path)
    
    # 모델에 가중치 적용
    model.load_state_dict(state_dict)
    print(f"모델이 {model_path}에서 로드되었습니다.")
    
    return model

def generate_bass_for_midi(input_midi_path, output_midi_path, model, device, beat_resolution=4):
    """MIDI 파일의 다른 트랙들에 대해 베이스 트랙 생성"""
    # MIDI 파일 로드
    midi_data = pretty_midi.PrettyMIDI(input_midi_path)
    
    # 베이스 트랙이 아닌 트랙들 찾기
    other_tracks = []
    for instrument in midi_data.instruments:
        # 베이스 트랙 제외 (program number 33-39는 베이스)
        if not ('bass' in instrument.name.lower() or (32 <= instrument.program <= 39)):
            other_tracks.append(instrument)
    
    if not other_tracks:
        print("베이스가 아닌 트랙을 찾을 수 없습니다.")
        return
    
    # 모든 트랙을 piano roll 형태로 변환
    other_pianorolls = []
    for track in other_tracks:
        piano_roll = track.get_piano_roll(fs=beat_resolution)
        piano_roll = (piano_roll > 0).astype(np.float32)
        other_pianorolls.append(piano_roll)
    
    # 모든 피아노롤을 같은 길이로 맞추기
    max_length = max([p.shape[1] for p in other_pianorolls])
    for i in range(len(other_pianorolls)):
        if other_pianorolls[i].shape[1] < max_length:
            padding = np.zeros((128, max_length - other_pianorolls[i].shape[1]), dtype=np.float32)
            other_pianorolls[i] = np.concatenate([other_pianorolls[i], padding], axis=1)
    
    # 모든 트랙 합치기
    combined_pianoroll = np.zeros_like(other_pianorolls[0])
    for roll in other_pianorolls:
        combined_pianoroll = np.logical_or(combined_pianoroll, roll).astype(np.float32)
    
    # 텐서로 변환 [seq_len, 128] -> [1, seq_len, 128]
    other_tensor = torch.from_numpy(combined_pianoroll.T).unsqueeze(0).to(device)
    
    # 모델을 평가 모드로 설정
    model.eval()
    
with torch.no_grad():
        # 초기 시작 토큰 생성
        seq_len = other_tensor.size(1)
        start_token = torch.zeros(1, 1, 128, device=device)
        bass_seq = start_token
        
        # 자기회귀(auto-regressive) 생성
        # seq_len-1로 변경 (중요: 길이 불일치 해결)
        for i in range(seq_len-1):
            # 현재 다른 트랙 입력 - 시퀀스 길이 불일치 방지
            current_input = other_tensor[:, :i+1] if i < seq_len-1 else other_tensor[:, :seq_len-1]
            
            # 다음 음표 예측
            pred = model(current_input, bass_seq)
            
            # 예측값의 마지막 시점 가져오기
            next_note = (torch.sigmoid(pred[:, -1:, :]) > 0.5).float()
            
            # 시퀀스에 추가
            bass_seq = torch.cat([bass_seq, next_note], dim=1)
        
        # 첫 번째 토큰(시작 토큰) 제외, 원래 길이와 맞추기
        bass_seq = bass_seq[:, 1:].cpu().numpy()[0]  # [seq_len-1, 128]
        
        # 만약 생성된 베이스 시퀀스가 원본 피아노롤보다 짧으면 패딩 추가
        if bass_seq.shape[0] < combined_pianoroll.shape[1]:
            padding_length = combined_pianoroll.shape[1] - bass_seq.shape[0]
            padding = np.zeros((padding_length, 128), dtype=np.float32)
            bass_seq = np.concatenate([bass_seq, padding], axis=0)
        # 반대로 길면 자르기
        elif bass_seq.shape[0] > combined_pianoroll.shape[1]:
            bass_seq = bass_seq[:combined_pianoroll.shape[1], :]
    
    # pianoroll에서 MIDI 노트로 변환
    bass_instrument = pretty_midi.Instrument(
        program=33,  # Acoustic Bass
        name="Generated Bass"
    )
    
    # 피아노롤에서 노트 추출
    for pitch in range(128):
        # 현재 피치의 활성화 위치 찾기
        active = np.where(bass_seq[:, pitch] > 0)[0]
        
        # 연속된 활성화를 노트로 변환
        if len(active) > 0:
            # 활성화된 포인트의 차이가 1보다 큰 지점 찾기 (노트 경계)
            note_boundaries = np.concatenate([[0], np.where(np.diff(active) > 1)[0] + 1, [len(active)]])
            
            for i in range(len(note_boundaries) - 1):
                start_idx = active[note_boundaries[i]]
                end_idx = active[note_boundaries[i+1] - 1] + 1 if note_boundaries[i+1] < len(active) else active[-1] + 1
                
                # MIDI 노트 생성
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start_idx / beat_resolution,
                    end=end_idx / beat_resolution
                )
                bass_instrument.notes.append(note)
    
    # 기존 MIDI 파일에 베이스 트랙 추가
    # 기존 베이스 트랙 제거
    midi_data.instruments = [inst for inst in midi_data.instruments 
                            if not ('bass' in inst.name.lower() or (32 <= inst.program <= 39))]
    
    # 새 베이스 트랙 추가
    midi_data.instruments.append(bass_instrument)
    
    # 변경된 MIDI 파일 저장
    midi_data.write(output_midi_path)
    print(f"생성된 베이스 트랙이 포함된 MIDI 파일이 {output_midi_path}에 저장되었습니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI 파일의 베이스 트랙 생성")
    parser.add_argument("--input", type=str, required=True, help="베이스 트랙을 추가할 입력 MIDI 파일 경로")
    parser.add_argument("--output", type=str, required=True, help="베이스 트랙이 추가된 출력 MIDI 파일 경로")
    parser.add_argument("--model", type=str, default="models/bass_generator_model.safetensors", help="학습된 모델 파일 경로")
    
    args = parser.parse_args()
    
    # GPU 사용 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 디바이스: {device}")
    
    # 모델 로드
    model = load_model(args.model, device)
    
    # 베이스 생성
    generate_bass_for_midi(args.input, args.output, model, device)