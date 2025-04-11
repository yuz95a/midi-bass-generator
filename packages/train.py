import torch
import torch.nn.functional as F
from safetensors.torch import save_file, load_file
import os
import math

import print_cuda_info
import midi_bass_dataset
import transformer


def train_model(model, train_loader, optimizer, epochs, device):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        batch_count = 0

        for other_tracks, bass_track in train_loader:
            # 데이터를 GPU로 이동
            other_tracks = other_tracks.to(device)  # [batch_size, seq_len, 128]
            bass_track = bass_track.to(device)      # [batch_size, seq_len, 128]

            optimizer.zero_grad()
            
            # 모델 예측 (Transformer)
            # bass_track[:, :-1]은 디코더 입력, bass_track[:, 1:]은 예측 타겟
            pred_bass = model(other_tracks, bass_track[:, :-1])  # [batch_size, seq_len-1, 128]
            
            # 손실 계산 - 차원 확인 및 조정
            # pred_bass: [batch_size, seq_len-1, 128]
            # bass_track[:, 1:]: [batch_size, seq_len-1, 128]
            
            # 이진 분류 문제로 처리 (각 음표가 활성화되어 있는지 여부)
            loss = F.binary_cross_entropy_with_logits(
                pred_bass, 
                bass_track[:, 1:],
                reduction='mean'
            )
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")


def generate_bass(model, other_tracks, device):
    model.eval()  # 평가 모드로 설정
    
    # 데이터를 GPU로 이동
    other_tracks = other_tracks.to(device)  # [batch_size, seq_len, 128]
    batch_size, seq_len, _ = other_tracks.shape
    
    with torch.no_grad():
        # 초기 시작 토큰 생성 (모든 음표가 꺼진 상태)
        start_token = torch.zeros(batch_size, 1, 128, device=device)
        bass_seq = start_token
        
         # Auto-regressive 방식으로 생성
        for i in range(seq_len):
            # 현재까지의 시퀀스로 다음 음표 예측
            pred = model(other_tracks[:, :i+1], bass_seq)
            next_note = pred[:, -1:, :]  # 마지막 예측값
            
            # 이진 분류로 처리 (threshold 0.5)
            next_note = (torch.sigmoid(next_note) > 0.5).float()
            
            # 시퀀스에 추가
            bass_seq = torch.cat([bass_seq, next_note], dim=1)
        
        return bass_seq[:, 1:]  # 시작 토큰 제외

def train_with_ewc_and_replay(model, current_loader, replay_buffer, optimizer, epochs, ewc, replay_batch_size=16):
    for epoch in range(epochs):
        for other_tracks, bass_tracks in current_loader:
            optimizer.zero_grad()
            
            # 현재 데이터 손실
            outputs = model(other_tracks, bass_tracks[:, :-1])
            current_loss = F.cross_entropy(outputs.flatten(0, 1), bass_tracks[:, 1:].flatten())
            
            # 리플레이 버퍼 손실
            replay_loss = 0
            replay_data = replay_buffer.get_samples(replay_batch_size)
            if replay_data is not None:
                replay_other, replay_bass = replay_data
                replay_outputs = model(replay_other, replay_bass[:, :-1])
                replay_loss = F.cross_entropy(replay_outputs.flatten(0, 1), replay_bass[:, 1:].flatten())
            
            # EWC 패널티
            ewc_loss = ewc.penalty(model)
            
            # 손실 합산
            loss = current_loss + replay_loss + ewc_loss
            
            loss.backward()
            optimizer.step()

def load_model_from_safetensors(model, load_path):
    state_dict = load_file(load_path)
    model.load_state_dict(state_dict)
    print(f"모델이 성공적으로 {load_path}에서 로드되었습니다.")
    
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print_cuda_info.printCUDAinfo()

    # 데이터 로더 생성
    train_loader, test_loader = midi_bass_dataset.create_dataloaders(
        batch_size=16,  # 필요에 따라 조정
        num_workers=4   # 필요에 따라 조정
    )

    # 모델 초기화 및 GPU로 이동
    model = transformer.MIDIBassGenerator(
        input_dim=128,     # MIDI 음높이 수
        hidden_dim=256,    # 중간 크기의 히든 차원
        num_layers=3,      # 중간 수준의 복잡성
        output_dim=128     # MIDI 음높이 수
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    # 모델 학습
    epochs = 10
    train_model(model, train_loader, optimizer, epochs, device)
    
    # 모델 저장
    models_dir = "models"  # 모델 저장 디렉토리
    os.makedirs(models_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    
    state_dict = model.state_dict()
    filename = "model1.safetensors"
    save_file(state_dict, os.path.join(models_dir, filename))
    print(f"모델이 성공적으로 {os.path.join(models_dir, filename)}에 저장되었습니다.")
    

#     """모델 로드"""
#     model = transformer.MIDIBassGenerator(
#         input_dim=128,
#         hidden_dim=256,
#         num_layers=3,
#         output_dim=128
#     ).to(device)
    
#     # safetensors 파일에서 state_dict 로드
#     state_dict = load_file(os.path.join("models", "model.safetensors"))
    
#     # 모델에 가중치 적용
#     model.load_state_dict(state_dict)
#     print(f"모델이 로드되었습니다.")

#     # 테스트 예측 (선택 사항)
# for other_tracks, bass_tracks in test_loader:
#     with torch.no_grad():
#         # 데이터를 GPU로 이동
#         other_tracks = other_tracks.to(device)
#         bass_tracks = bass_tracks.to(device)  # 명시적으로 bass_tracks도 device로 이동
        
#         # 한 배치만 사용
#         gen_bass = generate_bass(model, other_tracks, device)
#         print(f"생성된 베이스 트랙 형태: {gen_bass.shape}")
#         print(f"실제 베이스 트랙 형태: {bass_tracks.shape}")
        
#         # 디버깅을 위한 크기 확인
#         print(f"생성된 베이스 트랙 길이: {gen_bass.size(1)}")
#         print(f"원본 베이스 트랙 길이: {bass_tracks.size(1)}")
        
#         # 크기가 다른 경우 작은 쪽에 맞추기
#         min_length = min(gen_bass.size(1), bass_tracks[:, 1:].size(1))
        
#         # 정확도 계산 (실제와 생성된 베이스 트랙 비교)
#         actual_bass = bass_tracks[:, 1:1+min_length]  # 첫 음표 제외하고 크기 맞춤
#         gen_bass_trimmed = gen_bass[:, :min_length]   # 생성된 베이스 트랙 크기 맞춤
        
#         # 정확도 계산 전 형상 확인
#         print(f"정확도 계산용 생성 베이스 형태: {gen_bass_trimmed.shape}")
#         print(f"정확도 계산용 실제 베이스 형태: {actual_bass.shape}")
        
#         # 디버깅을 위한 장치 확인
#         print(f"생성 베이스 장치: {gen_bass_trimmed.device}")
#         print(f"실제 베이스 장치: {actual_bass.device}")
        
#         # 이제 크기가 맞춰졌으므로 정확도 계산
#         accuracy = torch.mean((gen_bass_trimmed == (actual_bass > 0.5).float()).float())
#         print(f"테스트 정확도: {accuracy.item():.4f}")
#     break