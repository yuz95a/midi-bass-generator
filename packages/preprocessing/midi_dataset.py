import torch
import pickle
import os

import midi_bass_dataset

def save_datasets(train_loader, test_loader, train_path="train_dataset.pt", test_path="test_dataset.pt"):
    """데이터로더의 데이터를 파일로 저장"""
    # 데이터 수집
    train_data = []
    test_data = []
    
    print("학습 데이터 수집 중...")    
    for other_tracks, bass_tracks in train_loader:
        train_data.append((other_tracks, bass_tracks))
    
    print("테스트 데이터 수집 중...")
    for other_tracks, bass_tracks in test_loader:
        test_data.append((other_tracks, bass_tracks))
    
    # 데이터 저장
    print(f"데이터 저장 중: {train_path}, {test_path}")
    torch.save(train_data, train_path)
    torch.save(test_data, test_path)
    print("데이터 저장 완료")

def load_datasets(train_path="train_dataset.pt", test_path="test_dataset.pt"):
    """저장된 데이터셋 파일 불러오기"""
    print(f"데이터 불러오는 중: {train_path}, {test_path}")
    train_data = torch.load(train_path)
    test_data = torch.load(test_path)
    print("데이터 불러오기 완료")
    return train_data, test_data

if __name__ == "__main__":
    # 사용 예시:
    # 1. 데이터 저장
    train_loader, test_loader = midi_bass_dataset.create_dataloaders(
        train_dir='./midi/train', 
        test_dir='./midi/test', 
        batch_size=64, 
        seq_length=64, 
        beat_resolution=4,
        num_workers=16
    )
    save_datasets(train_loader, test_loader)

    # 2. 데이터 불러오기
    # train_data, test_data = load_datasets()