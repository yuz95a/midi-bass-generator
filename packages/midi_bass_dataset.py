import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import glob

class MIDIBassDataset(Dataset):
    def __init__(self, midi_dir, seq_length=64, beat_resolution=4):
        """
        MIDI 파일에서 베이스 트랙과 다른 트랙들을 로드하는 데이터셋
        
        Args:
            midi_dir (str): MIDI 파일이 있는 디렉토리 경로
            seq_length (int): 시퀀스 길이 (16분음표 기준)
            beat_resolution (int): 한 박자당 분해능(resolution)
        """
        self.midi_dir = midi_dir
        self.seq_length = seq_length
        self.beat_resolution = beat_resolution
        self.midi_files = glob.glob(os.path.join(midi_dir, '**/*.midi'), recursive=True)
        
        # 유효하지 않은 MIDI 파일 필터링
        self.valid_files = []
        for file in self.midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(file)
                if len(midi_data.instruments) > 1:  # 적어도 2개 이상의 트랙 필요
                    self.valid_files.append(file)
            except Exception as e:
                print(f"파일 로드 중 오류 발생: {file}, 오류: {e}")
        
        print(f"로드된 유효한 MIDI 파일: {len(self.valid_files)}/{len(self.midi_files)}")
        
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        """
        MIDI 파일을 로드하고 베이스 트랙과 다른 트랙으로 분리
        
        Returns:
            tuple: (other_tracks_tensor, bass_track_tensor)
        """
        midi_path = self.valid_files[idx]
        
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
            
            # 베이스 트랙 찾기
            bass_track = None
            other_tracks = []
            
            for instrument in midi_data.instruments:
                # 베이스 트랙 식별 (program number 33-39는 베이스)
                is_bass = 'bass' in instrument.name.lower() or (32 <= instrument.program <= 39)
                
                # 첫 번째 발견된 베이스 트랙을 사용
                if is_bass and bass_track is None:
                    bass_track = instrument
                else:
                    other_tracks.append(instrument)
            
            # 모든 트랙이 piano roll 형태로 변환
            bass_pianoroll = bass_track.get_piano_roll(fs=self.beat_resolution)
            bass_pianoroll = (bass_pianoroll > 0).astype(np.float32)
            
            # 다른 트랙들 변환 및 합치기
            other_pianorolls = []
            for track in other_tracks:
                piano_roll = track.get_piano_roll(fs=self.beat_resolution)
                piano_roll = (piano_roll > 0).astype(np.float32)
                other_pianorolls.append(piano_roll)
            
            # 다른 트랙들 합치기 (OR 연산으로)
            if other_pianorolls:
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
            else:
                combined_pianoroll = np.zeros((128, self.seq_length), dtype=np.float32)
            
            # 길이 맞추기
            min_length = min(bass_pianoroll.shape[1], combined_pianoroll.shape[1])
            if min_length < self.seq_length:
                # 시퀀스가 너무 짧으면 패딩
                bass_pad = np.zeros((128, self.seq_length - min_length), dtype=np.float32)
                other_pad = np.zeros((128, self.seq_length - min_length), dtype=np.float32)
                
                bass_pianoroll = np.concatenate([bass_pianoroll[:, :min_length], bass_pad], axis=1)
                combined_pianoroll = np.concatenate([combined_pianoroll[:, :min_length], other_pad], axis=1)
            else:
                # 랜덤 오프셋으로 시퀀스 자르기
                if min_length > self.seq_length:
                    offset = np.random.randint(0, min_length - self.seq_length)
                    bass_pianoroll = bass_pianoroll[:, offset:offset+self.seq_length]
                    combined_pianoroll = combined_pianoroll[:, offset:offset+self.seq_length]
                else:
                    bass_pianoroll = bass_pianoroll[:, :self.seq_length]
                    combined_pianoroll = combined_pianoroll[:, :self.seq_length]
            
            # 텐서로 변환 (시퀀스 길이, 피치)로 변환
            bass_tensor = torch.from_numpy(bass_pianoroll.T)  # [seq_len, 128]
            other_tensor = torch.from_numpy(combined_pianoroll.T)  # [seq_len, 128]
            
            return other_tensor, bass_tensor
            
        except Exception as e:
            print(f"파일 처리 중 오류 발생: {midi_path}, 오류: {e}")
            # 오류 발생 시 빈 텐서 반환
            return torch.zeros((self.seq_length, 128)), torch.zeros((self.seq_length, 128))


def create_dataloaders(train_dir='./midi/train', 
                       test_dir='./midi/test', 
                       batch_size=32, 
                       seq_length=64, 
                       beat_resolution=4,
                       num_workers=4):
    """
    학습 및 테스트 데이터 로더 생성
    
    Args:
        train_dir (str): 학습 MIDI 파일 디렉토리
        test_dir (str): 테스트 MIDI 파일 디렉토리
        batch_size (int): 배치 크기
        seq_length (int): 시퀀스 길이
        beat_resolution (int): 한 박자당 분해능
        num_workers (int): 데이터 로딩 작업자 수
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # 데이터셋 생성
    train_dataset = MIDIBassDataset(train_dir, seq_length, beat_resolution)
    test_dataset = MIDIBassDataset(test_dir, seq_length, beat_resolution)
    
    # 데이터 로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders()
    
    # train 샘플 확인
    for other_tracks, bass_tracks in train_loader:
        print(f"train 샘플")
        print(f"다른 트랙 형태: {other_tracks.shape}")
        print(f"베이스 트랙 형태: {bass_tracks.shape}")
        break

    # test 샘플 확인
    for other_tracks, bass_tracks in test_loader:
        print(f"test 샘플")
        print(f"다른 트랙 형태: {other_tracks.shape}")
        print(f"베이스 트랙 형태: {bass_tracks.shape}")
        break