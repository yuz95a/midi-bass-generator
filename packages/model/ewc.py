import torch
import torch.nn as nn
import torch.nn.functional as F

class EWC:
    def __init__(self, model, dataloader, importance=1000):
        """
        EWC 초기화
        
        Args:
            model: 학습된 모델
            dataloader: 이전 데이터셋의 데이터 로더
            importance: EWC 패널티 중요도
        """
        self.model = model
        self.importance = importance
        
        # 현재 모델 파라미터 저장
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        
        # Fisher Information 계산을 위한 준비
        self.fisher = self._calculate_fisher(dataloader)
        
    def _calculate_fisher(self, dataloader):
        """Fisher Information Matrix 계산"""
        fisher = {n: torch.zeros_like(p) for n, p in self.params.items()}
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        # 데이터로더에서 샘플 추출
        for other_tracks, bass_tracks in dataloader:
            # 모델 그래디언트 초기화
            self.model.zero_grad()
            
            # 모델 출력
            outputs = self.model(other_tracks, bass_tracks[:, :-1])
            
            # 로그 확률 계산
            log_probs = F.log_softmax(outputs, dim=-1)
            labels = bass_tracks[:, 1:].flatten()
            
            # 네거티브 로그 라이클리후드 계산
            loss = F.nll_loss(log_probs.flatten(0, 1), labels)
            
            # 역전파
            loss.backward()
            
            # Fisher Information 누적
            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.pow(2).detach()
        
        # 데이터셋 크기로 나누어 평균화
        fisher = {n: (f / len(dataloader)) for n, f in fisher.items()}
        
        return fisher
    
    def penalty(self, model):
        """EWC 패널티 계산"""
        loss = 0
        for n, p in model.named_parameters():
            if n in self.fisher and p.requires_grad:
                # 현재 파라미터와 저장된 파라미터 간의 차이에 Fisher 정보를 가중치로 사용
                loss += (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        
        return self.importance * loss / 2

# EWC를 사용한 학습 함수
def train_with_ewc(model, new_loader, optimizer, epochs, ewc):
    """
    EWC를 사용하여 모델 학습
    
    Args:
        model: 학습할 모델
        new_loader: 새 데이터 로더
        optimizer: 옵티마이저
        epochs: 에폭 수
        ewc: EWC 인스턴스
    """
    for epoch in range(epochs):
        for other_tracks, bass_tracks in new_loader:
            optimizer.zero_grad()
            
            # 모델 예측
            outputs = model(other_tracks, bass_tracks[:, :-1])
            
            # 기본 손실 계산
            task_loss = F.cross_entropy(outputs.flatten(0, 1), bass_tracks[:, 1:].flatten())
            
            # EWC 패널티 추가
            ewc_loss = ewc.penalty(model)
            
            # 총 손실
            loss = task_loss + ewc_loss
            
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}, Task Loss: {task_loss.item()}, EWC Loss: {ewc_loss.item()}")

# 사용 예시
# 1. 첫 번째 데이터셋으로 학습
# 2. EWC 초기화
ewc = EWC(model, first_dataloader)
# 3. 새 데이터셋으로 학습
train_with_ewc(model, new_dataloader, optimizer, epochs, ewc)