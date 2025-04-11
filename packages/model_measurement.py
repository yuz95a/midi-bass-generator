import torch
import torch.nn as nn
import transformer
import os
import time
from torch.cuda.amp import autocast
import psutil
import gc

def measure_memory_usage(model, seq_length=64, batch_size=16, use_amp=False):
    """
    모델의 메모리 사용량을 측정합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 사용 가능한 CUDA 장치 목록 출력
    if torch.cuda.is_available():
        print(f"사용 가능한 CUDA 장치: {torch.cuda.device_count()}")
        print(f"현재 CUDA 장치: {torch.cuda.current_device()}")
        print(f"현재 CUDA 장치 이름: {torch.cuda.get_device_name(0)}")
    
    # 가비지 컬렉션 및 CUDA 캐시 비우기
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 초기 메모리 사용량 기록
    initial_memory = 0
    if torch.cuda.is_available():
        initial_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    print(f"초기 CUDA 메모리 사용량: {initial_memory:.2f} MB")
    
    # 모델을 장치로 이동
    model = model.to(device)
    
    # 모델 이동 후 메모리 사용량
    after_model_memory = 0
    if torch.cuda.is_available():
        after_model_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    print(f"모델 로드 후 CUDA 메모리 사용량: {after_model_memory:.2f} MB")
    print(f"모델 가중치가 사용하는 메모리: {after_model_memory - initial_memory:.2f} MB")
    
    # 임의의 입력 데이터 생성
    other_tracks = torch.rand(batch_size, seq_length, 128, device=device)
    bass_track = torch.rand(batch_size, seq_length, 128, device=device)
    
    # 포워드 패스 전 메모리 측정
    before_forward_memory = 0
    if torch.cuda.is_available():
        before_forward_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    # 추론(Inference) 메모리 측정
    model.eval()
    with torch.no_grad():
        # 혼합 정밀도 사용 여부에 따라 다르게 처리
        if use_amp:
            with autocast():
                output = model(other_tracks, bass_track[:, :-1])
        else:
            output = model(other_tracks, bass_track[:, :-1])
    
    # 추론 후 메모리 측정
    inference_memory = 0
    if torch.cuda.is_available():
        inference_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    print(f"추론 이후 CUDA 메모리 사용량: {inference_memory:.2f} MB")
    print(f"추론에 추가로 사용된 메모리: {inference_memory - before_forward_memory:.2f} MB")
    
    # 학습 메모리 측정
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # 옵티마이저 생성 후 메모리
    after_opt_memory = 0
    if torch.cuda.is_available():
        after_opt_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
    
    print(f"옵티마이저 생성 후 CUDA 메모리 사용량: {after_opt_memory:.2f} MB")
    print(f"옵티마이저가 추가로 사용하는 메모리: {after_opt_memory - inference_memory:.2f} MB")
    
    # 학습 단계에서의 메모리 사용량 측정
    # 먼저 그래디언트 초기화
    optimizer.zero_grad()
    
    # 혼합 정밀도 사용 여부에 따라 다르게 처리
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        with autocast():
            output = model(other_tracks, bass_track[:, :-1])
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output, bass_track[:, 1:], reduction='mean'
            )
        
        # 역전파
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(other_tracks, bass_track[:, :-1])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output, bass_track[:, 1:], reduction='mean'
        )
        
        # 역전파
        loss.backward()
        optimizer.step()
    
    # 학습 후 메모리 측정
    training_memory = 0
    if torch.cuda.is_available():
        training_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
    
    print(f"학습 이후 CUDA 메모리 사용량: {training_memory:.2f} MB")
    print(f"최대 CUDA 메모리 사용량: {peak_memory:.2f} MB")
    print(f"학습에 추가로 사용된 메모리: {training_memory - after_opt_memory:.2f} MB")
    
    # 총 메모리 변화 요약
    print("\n메모리 사용량 요약:")
    print(f"모델 가중치: {after_model_memory - initial_memory:.2f} MB")
    print(f"추론 활성화: {inference_memory - before_forward_memory:.2f} MB")
    print(f"옵티마이저 상태: {after_opt_memory - inference_memory:.2f} MB")
    print(f"역전파 및 그래디언트: {training_memory - after_opt_memory:.2f} MB")
    print(f"총 메모리 사용량: {peak_memory:.2f} MB")
    
    # 시스템 메모리 사용량 (CPU RAM)
    process = psutil.Process(os.getpid())
    cpu_memory = process.memory_info().rss / (1024**2)  # MB
    print(f"프로세스 CPU 메모리 사용량: {cpu_memory:.2f} MB")
    
    return {
        "model_params": after_model_memory - initial_memory,
        "inference_memory": inference_memory - before_forward_memory,
        "optimizer_memory": after_opt_memory - inference_memory,
        "backprop_memory": training_memory - after_opt_memory,
        "peak_memory": peak_memory,
        "cpu_memory": cpu_memory,
        "use_amp": use_amp
    }


def measure_inference_speed(model, seq_length=64, batch_size=16, use_amp=False, num_iterations=100):
    """
    모델 추론 속도를 측정합니다.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # 임의의 입력 데이터 생성
    other_tracks = torch.rand(batch_size, seq_length, 128, device=device)
    bass_track = torch.rand(batch_size, seq_length, 128, device=device)
    
    # 워밍업
    print("워밍업 중...")
    with torch.no_grad():
        for _ in range(10):
            if use_amp:
                with autocast():
                    _ = model(other_tracks, bass_track[:, :-1])
            else:
                _ = model(other_tracks, bass_track[:, :-1])
    
    # 속도 측정
    print(f"추론 속도 측정 중 ({num_iterations}회 반복)...")
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            if use_amp:
                with autocast():
                    _ = model(other_tracks, bass_track[:, :-1])
            else:
                _ = model(other_tracks, bass_track[:, :-1])
    
    end_time = time.time()
    
    # 결과 계산
    elapsed_time = end_time - start_time
    avg_time_per_batch = elapsed_time / num_iterations * 1000  # 밀리초 단위
    avg_time_per_sample = avg_time_per_batch / batch_size  # 샘플당 평균 시간
    
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    print(f"배치당 평균 시간: {avg_time_per_batch:.2f} ms")
    print(f"샘플당 평균 시간: {avg_time_per_sample:.2f} ms")
    
    return {
        "total_time": elapsed_time,
        "avg_time_per_batch": avg_time_per_batch,
        "avg_time_per_sample": avg_time_per_sample,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "use_amp": use_amp
    }


if __name__ == "__main__":
    # 기존 작은 모델 (약 5.6M 파라미터)
    small_model = transformer.MIDIBassGenerator(
        input_dim=128,
        hidden_dim=256,
        num_layers=3,
        nhead=8,
        output_dim=128
    )
    
    # 중간 크기 모델 (약 20M 파라미터)
    medium_model = transformer.MIDIBassGenerator(
        input_dim=128,
        hidden_dim=512,
        num_layers=6,
        nhead=8,
        output_dim=128
    )
    
    # 큰 모델 (약 50M 파라미터)
    large_model = transformer.MIDIBassGenerator(
        input_dim=128,
        hidden_dim=768,
        num_layers=7,
        nhead=12,
        output_dim=128
    )
    
    print("=" * 50)
    print("모델 파라미터 수 비교")
    print("=" * 50)
    small_params = sum(p.numel() for p in small_model.parameters() if p.requires_grad)
    medium_params = sum(p.numel() for p in medium_model.parameters() if p.requires_grad)
    large_params = sum(p.numel() for p in large_model.parameters() if p.requires_grad)
    
    print(f"작은 모델 (256차원, 3레이어): {small_params:,} 파라미터")
    print(f"중간 모델 (512차원, 6레이어): {medium_params:,} 파라미터")
    print(f"큰 모델 (1024차원, 8레이어): {large_params:,} 파라미터")