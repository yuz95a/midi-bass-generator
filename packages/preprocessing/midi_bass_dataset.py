import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pretty_midi
import glob

class MIDIBassDataset(Dataset):
    def __init__(self, midi_dir, seq_length=64, beat_resolution=4):
        """
        MIDI 파일에서 베이스 트랙과 다른 악기 트랙을 로드하는 데이터셋
        
        Args:
            midi_dir (str): MIDI 파일이 있는 디렉토리 경로
            seq_length (int): 몇 개의 시간 단위를 한 묶음으로 볼 것인지 나타냄
            beat_resolution (int): 한 박자를 몇 개의 시간 단위로 나타낼 것인지 나타냄 1은 한 박, 2는 반 박, 4는 1/4박
        """
        self.midi_dir = midi_dir
        self.seq_length = seq_length
        self.beat_resolution = beat_resolution
        self.midi_files = glob.glob(os.path.join(midi_dir, '**/*.midi'), recursive=True)
        
        # 유효하지 않은 MIDI 파일 필터링
        # 트랙이 2개 이상인 것만 필터링
        self.valid_files = []
        for file in self.midi_files:
            try:
                midi_data = pretty_midi.PrettyMIDI(file)
                if len(midi_data.instruments) > 1:
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
            
            ## 베이스 트랙 찾기
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
            
            ## 모든 트랙을 piano roll로 변환
            # velocity 값을 가진 piano roll 생성
            # 베이스 트랙 변환
            bass_pianoroll = bass_track.get_piano_roll(fs=self.beat_resolution)
            # 이진 piano roll로 변환 (0 또는 1)
            bass_pianoroll = (bass_pianoroll > 0).astype(np.float32)
            
            # 다른 악기 트랙 변환
            other_pianorolls = []
            for track in other_tracks:
                piano_roll = track.get_piano_roll(fs=self.beat_resolution)
                # 이진화 적용
                piano_roll = (piano_roll > 0).astype(np.float32)
                other_pianorolls.append(piano_roll)
            
            # 다른 트랙들 합치기 (최대값 사용)
            if other_pianorolls:
                # 모든 피아노롤을 같은 길이로 맞추기
                # piano roll[pitch:128, time_steps:any]
                max_length = max([p.shape[1] for p in other_pianorolls])
                for i in range(len(other_pianorolls)):
                    if other_pianorolls[i].shape[1] < max_length:
                        padding = np.zeros((128, max_length - other_pianorolls[i].shape[1]), dtype=np.float32)
                        other_pianorolls[i] = np.concatenate([other_pianorolls[i], padding], axis=1)
                
                # 모든 트랙 합치기 (최대값 사용)
                combined_pianoroll = np.zeros_like(other_pianorolls[0])
                for roll in other_pianorolls:
                    combined_pianoroll = np.maximum(combined_pianoroll, roll)
            else:
                raise Exception("다른 악기 트랙 없음")
            
            # 베이스 트랙과 다른 음악 트랙 길이를 seq_length로 맞추기
            # bass_pianoroll[pitch:128, time_steps:seq_length]
            # combined_pianoroll[0][pitch:128, time_steps:seq_length]
            min_length = min(bass_pianoroll.shape[1], combined_pianoroll.shape[1])
            # 로드한 트랙이 지정한 시퀀스 길이보다 짧으면 패딩
            if min_length < self.seq_length:
                bass_pad = np.zeros((128, self.seq_length - min_length), dtype=np.float32)
                other_pad = np.zeros((128, self.seq_length - min_length), dtype=np.float32)
                # 둘 중 짧은 것에 길이를 맞춰 자른 후 패딩
                bass_pianoroll = np.concatenate([bass_pianoroll[:, :min_length], bass_pad], axis=1)
                combined_pianoroll = np.concatenate([combined_pianoroll[:, :min_length], other_pad], axis=1)
            else:
                # 랜덤 오프셋으로 시퀀스 자르기
                if min_length > self.seq_length:
                    max_attempts = 100  # 최대 시도 횟수
                    found_good_sequence = False
                    
                    for attempt in range(max_attempts):
                        offset = np.random.randint(0, min_length - self.seq_length)
                        temp_bass = bass_pianoroll[:, offset:offset+self.seq_length]
                        temp_combined = combined_pianoroll[:, offset:offset+self.seq_length]
                        
                        # 베이스 piano roll에서 0이 아닌 값의 비율 계산
                        non_zero_ratio = np.count_nonzero(temp_bass) / (temp_bass.shape[0] * temp_bass.shape[1])
                        
                        # 0이 아닌 값이 전체의 50% 이상이면 선택
                        if non_zero_ratio >= 0.25:
                            bass_pianoroll = temp_bass
                            combined_pianoroll = temp_combined
                            found_good_sequence = True
                            break
                    
                    # 적합한 시퀀스를 찾지 못했다면 마지막 시도한 것 사용
                    if not found_good_sequence:
                        bass_pianoroll = temp_bass
                        combined_pianoroll = temp_combined
                else:
                    bass_pianoroll = bass_pianoroll[:, :self.seq_length]
                    combined_pianoroll = combined_pianoroll[:, :self.seq_length]
            
            # 텐서로 변환 [time_steps:seq_length, pitch:128]
            bass_tensor = torch.from_numpy(bass_pianoroll.T)
            other_tensor = torch.from_numpy(combined_pianoroll.T)
            
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
    train_loader, test_loader = create_dataloaders(
        train_dir='./midi/train', 
        test_dir='./midi/test', 
        batch_size=64, 
        seq_length=64, 
        beat_resolution=4,
        num_workers=16
    )
    
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