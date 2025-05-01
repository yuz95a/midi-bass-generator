import torch
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.amp import GradScaler, autocast
from safetensors.torch import save_file, load_file
import os
import math

import print_cuda_info
import midi_bass_dataset
import midi_dataset
import transformer


def train_model(model, train_loader, test_loader, optimizer, scheduler, epochs, device):
    # 혼합 정밀도 학습을 위한 scaler 초기화
    scaler = GradScaler('cuda')

    for epoch in range(epochs):
        model.train()

        train_total_loss = 0.0  # 이름 변경
        train_batch_count = 0   # 이름 변경

        # other_tracks[batch_size, seq_length, pitch]
        # bass_track[batch_size, seq_length, pitch]
        for other_tracks, bass_track in train_loader:
            # 데이터를 GPU로 이동하고 타입을 명시함
            other_tracks = other_tracks.to(device).float()
            bass_track = bass_track.to(device).float()

            optimizer.zero_grad()
            
            # 혼합 정밀도를 위한 autocast 적용
            with autocast('cuda'):
                # 모델 예측 (Transformer)
                # bass_track[:, :-1]은 디코더 입력, bass_track[:, 1:]은 예측 타겟
                pred_bass = model(other_tracks, bass_track[:, :-1])
                
                # 손실 계산
                loss = loss_function(pred_bass, bass_track[:, 1:, :])

                if not torch.isfinite(loss):
                    print(f"경고: 손실 값이 무한대 또는 NaN입니다: {loss.item()}")
                    continue
            
            # 스케일러를 사용하여 역전파 수행
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += loss.item()
            train_batch_count += 1
            
        scheduler.step()

        train_avg_loss = train_total_loss / train_batch_count
        print(f"Epoch {epoch+1}, Average Train Loss: {train_avg_loss:.6f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 테스트 데이터에 대한 평가
        model.eval()
        test_total_loss = 0.0  # 새로운 변수 생성
        test_batch_count = 0    # 새로운 변수 생성
        
        for other_tracks, bass_track in test_loader:
            # 데이터를 GPU로 이동하고 타입을 명시함
            other_tracks = other_tracks.to(device).float()
            bass_track = bass_track.to(device).float()
            with torch.no_grad():
                pred_bass = model(other_tracks, bass_track[:, :-1])
                loss = loss_function(pred_bass, bass_track[:, 1:, :])

                if not torch.isfinite(loss):
                    print(f"경고: 손실 값이 무한대 또는 NaN입니다: {loss.item()}")
                    continue
                    
            test_total_loss += loss.item()   
            test_batch_count += 1
            
        # 테스트 평균 손실 계산
        test_avg_loss = test_total_loss / test_batch_count
        print(f"Epoch {epoch+1}, Average Test Loss: {test_avg_loss:.6f}")

def loss_function(pred_bass, real_bass, temperature=10.0):
    """
    pred_bass: (B, T, N), 0~1 사이 확률값
    real_bass: (B, T, N), 0 또는 1 이진값
    """
    soft_binary = torch.sigmoid(temperature * (pred_bass - 0.5))

    miss_penalty = 0.5 * real_bass * (1.0 - soft_binary)
    false_positive_penalty = 2.0 * (1.0 - real_bass) * soft_binary

    loss = miss_penalty + false_positive_penalty
    return loss.mean().float()

def generate_bass(model, other_tracks, device, temperature=1.0):
    model.eval()  # 평가 모드로 설정
    
    # 데이터를 GPU로 이동
    other_tracks = other_tracks.to(device).float()
    batch_size, seq_len, _ = other_tracks.shape
    
    with torch.no_grad():
        # 초기 시작 토큰 생성 (모든 음표가 꺼진 상태)
        start_token = torch.zeros(batch_size, 1, 128, device=device).float()
        bass_seq = start_token
        
        # Auto-regressive 방식으로 생성
        for i in range(seq_len):
            # 현재까지의 시퀀스로 다음 음표 예측
            # other_tracks[batch_size, seq_len, 128]
            pred = model(other_tracks[:, :i+1, :], bass_seq)
            next_note = pred[:, -1:, :]  # 마지막 예측값
            
            # 샘플링할 때 온도 조절 적용 (높은 온도는 더 무작위적인 샘플링)
            if temperature != 1.0:
                next_note = next_note / temperature

            # 정규화된 값으로 유지 (0과 1 사이의 값으로 클리핑)
            next_note = torch.clamp(next_note, 0.0, 1.0)
            
            # 시퀀스에 추가
            bass_seq = torch.cat([bass_seq, next_note], dim=1)
        
        return bass_seq[:, 1:]  # 시작 토큰 제외

def train_with_ewc_and_replay(model, current_loader, replay_buffer, optimizer, epochs, ewc, replay_batch_size=16):
    for epoch in range(epochs):
        for other_tracks, bass_tracks in current_loader:
            optimizer.zero_grad()
            
            # 현재 데이터 손실
            outputs = model(other_tracks, bass_tracks[:, :-1])
            current_loss = F.mse_loss(outputs, bass_tracks[:, 1:])
            
            # 리플레이 버퍼 손실
            replay_loss = 0
            replay_data = replay_buffer.get_samples(replay_batch_size)
            if replay_data is not None:
                replay_other, replay_bass = replay_data
                replay_outputs = model(replay_other, replay_bass[:, :-1])
                replay_loss = F.mse_loss(replay_outputs, replay_bass[:, 1:])
            
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

def train_with_curriculum(model, train_loader, test_loader, optimizer, scheduler, epochs, device):
    """
    커리큘럼 학습을 적용한 훈련 함수
    점진적으로 시퀀스 길이를 늘리고 Teacher forcing을 줄임
    Teacher forcing과 Auto-regressive 생성 모두에 대한 테스트 손실을 평가
    """
    scaler = GradScaler('cuda')
    
    # 커리큘럼 설정
    curriculum_stages = [
        {"seq_length": 16, "tf_ratio": 0.9, "epochs": 10},
        {"seq_length": 32, "tf_ratio": 0.7, "epochs": 15},
        {"seq_length": 48, "tf_ratio": 0.5, "epochs": 15},
        {"seq_length": 64, "tf_ratio": 0.3, "epochs": 10}
    ]
    
    current_epoch = 0
    
    for stage in curriculum_stages:
        seq_length = stage["seq_length"]
        tf_ratio = stage["tf_ratio"]
        stage_epochs = stage["epochs"]
        
        print(f"\n시작: 커리큘럼 스테이지 - 시퀀스 길이={seq_length}, TF 비율={tf_ratio}")
        
        for epoch in range(stage_epochs):
            # ---------- 학습 단계 ----------
            model.train()
            train_tf_loss = 0.0
            train_ar_loss = 0.0
            train_total_loss = 0.0
            train_tf_count = 0
            train_ar_count = 0
            train_batch_count = 0
            
            for other_tracks, bass_track in train_loader:
                # 현재 커리큘럼 스테이지에 맞는 시퀀스 길이로 자르기
                other_tracks = other_tracks[:, :seq_length, :].to(device).float()
                bass_track = bass_track[:, :seq_length, :].to(device).float()
                
                optimizer.zero_grad()
                
                with autocast('cuda'):
                    # Teacher forcing 적용 여부 결정
                    use_tf = torch.rand(1).item() < tf_ratio
                    
                    if use_tf:
                        # Teacher forcing 모드
                        pred_bass = model(other_tracks, bass_track[:, :-1])
                        loss = loss_function(pred_bass, bass_track[:, 1:, :], temperature=5.0)
                        train_tf_loss += loss.item()
                        train_tf_count += 1
                    else:
                        # Auto-regressive 생성 모드
                        batch_size = bass_track.shape[0]
                        start_token = torch.zeros(batch_size, 1, 128, device=device).float()
                        generated = start_token
                        
                        for i in range(seq_length - 1):
                            pred = model(other_tracks[:, :i+1, :], generated)
                            next_note = pred[:, -1:, :]
                            generated = torch.cat([generated, next_note], dim=1)
                        
                        generated = generated[:, 1:]  # 시작 토큰 제외
                        loss = loss_function(generated, bass_track[:, 1:, :], temperature=5.0)
                        train_ar_loss += loss.item()
                        train_ar_count += 1
                
                # 기울기 계산 및 업데이트
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                
                train_total_loss += loss.item()
                train_batch_count += 1
            
            scheduler.step()
            
            # 학습 손실 평균 계산
            train_avg_loss = train_total_loss / train_batch_count
            train_tf_avg_loss = train_tf_loss / max(train_tf_count, 1)  # 0으로 나누기 방지
            train_ar_avg_loss = train_ar_loss / max(train_ar_count, 1)  # 0으로 나누기 방지
            
            # ---------- 테스트 단계 ----------
            model.eval()
            
            # Teacher forcing 평가
            test_tf_loss = 0.0
            test_tf_batch_count = 0
            
            # Auto-regressive 평가
            test_ar_loss = 0.0
            test_ar_batch_count = 0
            
            with torch.no_grad():
                for other_tracks, bass_track in test_loader:
                    # 현재 커리큘럼 스테이지에 맞는 시퀀스 길이로 자르기
                    other_tracks = other_tracks[:, :seq_length, :].to(device).float()
                    bass_track = bass_track[:, :seq_length, :].to(device).float()
                    
                    # 1. Teacher forcing 모드 평가
                    pred_bass = model(other_tracks, bass_track[:, :-1])
                    tf_loss = loss_function(pred_bass, bass_track[:, 1:, :], temperature=5.0)
                    test_tf_loss += tf_loss.item()
                    test_tf_batch_count += 1
                    
                    # 2. Auto-regressive 모드 평가
                    batch_size = bass_track.shape[0]
                    start_token = torch.zeros(batch_size, 1, 128, device=device).float()
                    generated = start_token
                    
                    for i in range(seq_length - 1):
                        pred = model(other_tracks[:, :i+1, :], generated)
                        next_note = pred[:, -1:, :]
                        generated = torch.cat([generated, next_note], dim=1)
                    
                    generated = generated[:, 1:]  # 시작 토큰 제외
                    ar_loss = loss_function(generated, bass_track[:, 1:, :], temperature=5.0)
                    test_ar_loss += ar_loss.item()
                    test_ar_batch_count += 1
            
            # 테스트 손실 평균 계산
            test_tf_avg_loss = test_tf_loss / test_tf_batch_count
            test_ar_avg_loss = test_ar_loss / test_ar_batch_count
            
            # 정보 출력
            print(f"Epoch {current_epoch+1}/{epochs}, 스테이지: 시퀀스 길이={seq_length}")
            print(f"  학습 손실: 전체={train_avg_loss:.6f}, TF={train_tf_avg_loss:.6f}, AR={train_ar_avg_loss:.6f}")
            print(f"  테스트 손실: TF={test_tf_avg_loss:.6f}, AR={test_ar_avg_loss:.6f}")
            print(f"  학습률: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Teacher forcing과 Auto-regressive 간 차이 계산
            train_diff = train_tf_avg_loss - train_ar_avg_loss
            test_diff = test_tf_avg_loss - test_ar_avg_loss
            print(f"  TF-AR 차이: 학습={train_diff:.6f}, 테스트={test_diff:.6f}")
            
            # Auto-regressive 생성의 추가 분석 (샘플 몇 개만)
            if current_epoch % 2 == 0:  # 2 에폭마다 상세 분석
                for other_tracks, bass_track in test_loader:
                    other_tracks = other_tracks[:2, :seq_length, :].to(device).float()  # 첫 2개 샘플만
                    bass_track = bass_track[:2, :seq_length, :].to(device).float()
                    
                    with torch.no_grad():
                        # Auto-regressive 생성
                        batch_size = other_tracks.shape[0]
                        start_token = torch.zeros(batch_size, 1, 128, device=device).float()
                        generated = start_token
                        
                        for i in range(seq_length - 1):
                            pred = model(other_tracks[:, :i+1, :], generated)
                            next_note = pred[:, -1:, :]
                            generated = torch.cat([generated, next_note], dim=1)
                        
                        generated = generated[:, 1:]  # 시작 토큰 제외
                        
                        # 생성된 베이스와 실제 베이스 비교 분석
                        gen_notes_per_step = generated.sum(dim=2).mean().item()
                        real_notes_per_step = bass_track[:, 1:].sum(dim=2).mean().item()
                        
                        # 결과 출력
                        print(f"  생성 분석: 생성된 음표/스텝={gen_notes_per_step:.4f}, 실제 음표/스텝={real_notes_per_step:.4f}")
                    
                    # 첫 샘플만 분석
                    break
            
            current_epoch += 1
            
            # 일정 주기로 모델 저장
            if current_epoch % 5 == 0 or (epoch == stage_epochs - 1):  # 5에폭마다 또는 스테이지 마지막에
                state_dict = model.state_dict()
                filename = f"model_epoch{current_epoch}_seq{seq_length}.safetensors"
                save_file(state_dict, os.path.join("models", filename))
                print(f"모델이 {filename}에 저장되었습니다.")
                
                # AR 모드로 생성된 샘플도 저장 (선택적)
                # 이 부분은 실제 MIDI 파일 생성 기능이 있다면 추가

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print_cuda_info.printCUDAinfo()

    # # 데이터 로더 생성
    # train_loader, test_loader = midi_bass_dataset.create_dataloaders(
    #     train_dir = './midi/train', 
    #     test_dir = './midi/test', 
    #     batch_size = 64, 
    #     seq_length = 64, 
    #     beat_resolution = 4,
    #     num_workers = 16
    # )

    train_loader, test_loader = midi_dataset.load_datasets()

    # 50M 파라미터 규모 모델
    model = transformer.MIDIBassGenerator(
        input_dim = 128,      # MIDI 음높이 수
        hidden_dim = 512,     # 히든 차원
        num_layers = 4,       # 레이어 수
        nhead = 8,            # 어텐션 헤드 수, 히든 차원에서 나누어 떨어질 수 있는 수
        output_dim = 128      # MIDI 음높이 수
    ).to(device).float()

    # 모델 파라미터 수 계산 및 출력
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"모델의 총 파라미터 수: {total_params:,}")
    print(f"모델 가중치 메모리 사용량 (FP32): {total_params * 4 / (1024**2):.2f} MB")
    print(f"모델 가중치 메모리 사용량 (FP16): {total_params * 2 / (1024**2):.2f} MB")

    # AdamW 옵티마이저 및 스케줄러
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0001,
        weight_decay=0.01,
        betas=(0.9, 0.98)
    )

    # 학습률 스케줄러 추가
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=300,  # 에폭 수에 맞게 조정
        eta_min=1e-6
    )

    # 모델 학습
    epochs = 50
    # train_model(model, train_loader, test_loader, optimizer, scheduler, epochs, device)

    
    # # 모델 저장
    # models_dir = "models"  # 모델 저장 디렉토리
    # os.makedirs(models_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    
    # state_dict = model.state_dict()
    # filename = "model3.safetensors"
    # save_file(state_dict, os.path.join(models_dir, filename))
    # print(f"모델이 성공적으로 {os.path.join(models_dir, filename)}에 저장되었습니다.")

    train_with_curriculum(model, train_loader, test_loader, optimizer, scheduler, epochs, device)
