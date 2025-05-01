import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

from midi_dataset import load_datasets

class BassGeneratorLSTM(nn.Module):
    def __init__(self, input_size=128, hidden_size=512, num_layers=2, output_size=128, dropout=0.5):
        """
        다른 악기의 피아노 롤을 입력으로 받아 베이스 피아노 롤을 생성하는 LSTM 모델
        
        Args:
            input_size (int): 입력 특성 차원 (피아노 롤에서 음높이 수, 기본값 128)
            hidden_size (int): LSTM 은닉층 차원
            num_layers (int): LSTM 레이어 수
            output_size (int): 출력 특성 차원 (피아노 롤에서 음높이 수, 기본값 128)
            dropout (float): 드롭아웃 비율
        """
        super(BassGeneratorLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM 레이어
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 출력 레이어
        # 양방향 LSTM이므로 hidden_size * 2
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()  # 이진 출력을 위한 시그모이드 활성화 함수
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): 입력 데이터 [batch_size, seq_length, input_size]
            
        Returns:
            Tensor: 출력 데이터 [batch_size, seq_length, output_size]
        """
        # LSTM 출력
        lstm_out, _ = self.lstm(x)
        
        # 출력 레이어
        output = self.fc(lstm_out)
        
        return output
    
    def generate(self, input_sequence, threshold=0.5):
        """
        주어진 입력 시퀀스에 대해 베이스 트랙 생성
        
        Args:
            input_sequence (Tensor): 입력 시퀀스 [batch_size, seq_length, input_size]
            threshold (float): 음표 활성화 임계값
            
        Returns:
            Tensor: 생성된 베이스 트랙 [batch_size, seq_length, output_size]
        """
        self.eval()  # 평가 모드로 전환
        with torch.no_grad():
            outputs = self(input_sequence)
            # 임계값 적용하여 이진화
            outputs = (outputs > threshold).float()
        return outputs


def train_model(model, train_loader, test_loader, device, 
                num_epochs=50, learning_rate=0.001, weight_decay=1e-5,
                save_dir='./models'):
    """
    모델 학습 함수
    
    Args:
        model (nn.Module): 학습할 모델
        train_loader (DataLoader): 학습 데이터 로더
        test_loader (DataLoader): 테스트 데이터 로더
        device (torch.device): 학습 디바이스 (CPU/GPU)
        num_epochs (int): 학습 에폭 수
        learning_rate (float): 학습률
        weight_decay (float): 가중치 감쇠 (L2 정규화)
        save_dir (str): 모델 저장 디렉토리
        
    Returns:
        dict: 학습 히스토리 (loss, accuracy 등)
    """
    # 손실 함수 및 옵티마이저 초기화
    criterion = nn.BCELoss()  # 이진 교차 엔트로피 손실
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 학습률 스케줄러
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 학습 결과 저장을 위한 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 학습 히스토리 기록
    history = {
        'train_loss': [],
        'test_loss': [],
        'train_accuracy': [],
        'test_accuracy': []
    }
    
    best_test_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        model.train()  # 학습 모드로 전환
        train_loss = 0.0
        train_accuracy = 0.0
        
        # 학습 데이터로 모델 학습
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for other_tracks, bass_tracks in progress_bar:
            # 데이터를 지정된 디바이스로 이동
            other_tracks = other_tracks.to(device)
            bass_tracks = bass_tracks.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(other_tracks)
            
            # 손실 계산
            loss = criterion(outputs, bass_tracks)
            
            # Backward pass 및 최적화
            loss.backward()
            optimizer.step()
            
            # 배치 손실 누적
            train_loss += loss.item()
            
            # 정확도 계산 (임계값 0.5 적용)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == bass_tracks).float().mean().item()
            train_accuracy += accuracy
            
            # 프로그레스 바 업데이트
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
        
        # 에폭 평균 손실 및 정확도 계산
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader)
        
        # 테스트 데이터로 모델 평가
        model.eval()  # 평가 모드로 전환
        test_loss = 0.0
        test_accuracy = 0.0
        
        with torch.no_grad():
            progress_bar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
            for other_tracks, bass_tracks in progress_bar:
                # 데이터를 지정된 디바이스로 이동
                other_tracks = other_tracks.to(device)
                bass_tracks = bass_tracks.to(device)
                
                # Forward pass
                outputs = model(other_tracks)
                
                # 손실 계산
                loss = criterion(outputs, bass_tracks)
                
                # 테스트 손실 누적
                test_loss += loss.item()
                
                # 정확도 계산 (임계값 0.5 적용)
                predictions = (outputs > 0.5).float()
                accuracy = (predictions == bass_tracks).float().mean().item()
                test_accuracy += accuracy
                
                # 프로그레스 바 업데이트
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{accuracy:.4f}'
                })
        
        # 에폭 평균 테스트 손실 및 정확도 계산
        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        
        # 학습률 스케줄러 업데이트
        scheduler.step(test_loss)
        
        # 현재 학습 상태 저장
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_accuracy'].append(test_accuracy)
        
        # 에폭 결과 출력
        end_time = time.time()
        epoch_time = end_time - start_time
        print(f'Epoch {epoch+1}/{num_epochs} - '
              f'Time: {epoch_time:.2f}s - '
              f'Train Loss: {train_loss:.4f} - '
              f'Test Loss: {test_loss:.4f} - '
              f'Train Acc: {train_accuracy:.4f} - '
              f'Test Acc: {test_accuracy:.4f}')
        
        # 최고 성능 모델 저장
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'test_loss': test_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f'모델 저장됨: {os.path.join(save_dir, "best_model.pth")}')
        
        # 마지막 모델 저장
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
        }, os.path.join(save_dir, 'last_model.pth'))
    
    # 학습 커브 시각화
    plt.figure(figsize=(12, 5))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_accuracy'], label='Train Accuracy')
    plt.plot(history['test_accuracy'], label='Test Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.show()
    
    return history


def load_model(model, model_path, device):
    """
    저장된 모델 로드 함수
    
    Args:
        model (nn.Module): 로드할 모델 인스턴스
        model_path (str): 모델 파일 경로
        device (torch.device): 모델을 로드할 디바이스
        
    Returns:
        nn.Module: 로드된 모델
    """
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'모델 로드됨 (에폭: {checkpoint["epoch"]}, 테스트 손실: {checkpoint["test_loss"]:.4f})')
    return model


def convert_to_midi(piano_roll, output_path, fs=4, program=32):
    """
    피아노 롤을 MIDI 파일로 변환
    
    Args:
        piano_roll (numpy.ndarray): 피아노 롤 [time_steps, pitch]
        output_path (str): 출력 MIDI 파일 경로
        fs (int): 초당 프레임 수
        program (int): MIDI 프로그램 번호 (0-127), 32는 베이스
    """
    import pretty_midi
    
    # PrettyMIDI 객체 생성
    midi = pretty_midi.PrettyMIDI()
    
    # 악기 추가
    instrument = pretty_midi.Instrument(program=program)
    
    # 피아노 롤 형태의 데이터를 Note 객체로 변환
    piano_roll = piano_roll.T  # [pitch, time]에서 [time, pitch]로 변환
    
    # 활성화된 노트를 찾기
    for pitch in range(piano_roll.shape[0]):
        # 피아노 롤에서 음표의 시작과 끝 인덱스 찾기
        note_on = False
        start_time = 0
        
        for time_idx, is_active in enumerate(piano_roll[pitch]):
            if is_active and not note_on:
                # 노트 시작
                note_on = True
                start_time = time_idx / fs
            elif (not is_active or time_idx == piano_roll.shape[1] - 1) and note_on:
                # 노트 종료 또는 마지막 프레임
                end_time = time_idx / fs
                # 노트 추가
                note = pretty_midi.Note(
                    velocity=100,  # 기본 볼륨
                    pitch=pitch,
                    start=start_time,
                    end=end_time
                )
                instrument.notes.append(note)
                note_on = False
    
    # 악기를 MIDI 파일에 추가
    midi.instruments.append(instrument)
    
    # MIDI 파일 저장
    midi.write(output_path)
    print(f'MIDI 파일 저장됨: {output_path}')


if __name__ == "__main__":
    # 설정
    batch_size = 64
    seq_length = 64
    beat_resolution = 4
    
    train_loader, test_loader = load_datasets()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'사용 디바이스: {device}')
    
    # 모델 초기화
    model = BassGeneratorLSTM(
        input_size=128,
        hidden_size=512,
        num_layers=2,
        output_size=128,
        dropout=0.5
    ).to(device)
    
    # 모델 구조 출력
    print(model)
    
    # 모델 학습
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        num_epochs=50,
        learning_rate=0.001,
        weight_decay=1e-5,
        save_dir='./models/bass_generator'
    )
    
    # 모델 평가 (테스트 샘플로 베이스 트랙 생성)
    model.eval()
    
    # 샘플 데이터 가져오기
    for other_tracks, bass_tracks in test_loader:
        # 몇 개의 샘플만 선택
        num_samples = min(3, other_tracks.shape[0])
        other_tracks = other_tracks[:num_samples].to(device)
        bass_tracks = bass_tracks[:num_samples].to(device)
        
        # 베이스 트랙 생성
        with torch.no_grad():
            generated_bass = model.generate(other_tracks)
        
        # CPU로 이동하고 넘파이 배열로 변환
        other_tracks = other_tracks.cpu().numpy()
        bass_tracks = bass_tracks.cpu().numpy()
        generated_bass = generated_bass.cpu().numpy()
        
        # 샘플별로 생성 결과 저장 및 시각화
        for i in range(num_samples):
            # MIDI 파일로 저장
            os.makedirs('./generated', exist_ok=True)
            
            # 원본 다른 트랙
            convert_to_midi(
                other_tracks[i], 
                f'./generated/sample_{i+1}_other_tracks.midi',
                fs=beat_resolution,
                program=0  # 피아노
            )
            
            # 실제 베이스 트랙
            convert_to_midi(
                bass_tracks[i], 
                f'./generated/sample_{i+1}_real_bass.midi',
                fs=beat_resolution,
                program=32  # 베이스
            )
            
            # 생성된 베이스 트랙
            convert_to_midi(
                generated_bass[i], 
                f'./generated/sample_{i+1}_generated_bass.midi',
                fs=beat_resolution,
                program=32  # 베이스
            )
            
            print(f'샘플 {i+1} 저장 완료')
        
        # 몇 개의 샘플만 처리
        break
    
    print('완료!')