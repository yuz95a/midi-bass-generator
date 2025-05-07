import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.utils.data import DataLoader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model) # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # [d_model // 2]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.d_model = d_model
        self.max_len = max_len
        
    def forward(self, x):
        '''
        x [batch, seq_len, d_model] [32, 256, 512]
        pe [unsqueeze, max_len, d_model] [1, 256, 512]
        '''
        # 차원 확인
        seq_len = x.size(1)
        if seq_len > self.max_len:
            # 시퀀스 길이가 너무 길면 경고 출력
            print(f"Warning: Input sequence length ({seq_len}) is longer than positional encoding max_len ({self.max_len})")
            # max_len 길이까지만 포지셔널 인코딩 적용
            x[:, :self.max_len] = x[:, :self.max_len] + self.pe[:, :self.max_len]
            return x
        
        # 포지셔널 인코딩
        x = x + self.pe[:, :seq_len]
        return x

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 src_seq_len=256, tgt_seq_len=64):
        super(TransformerModel, self).__init__()
        
        # feature 시퀀스 길이, label 시퀀스 길이
        self.src_seq_len = src_seq_len
        self.tgt_seq_len = tgt_seq_len
        
        # feature, label 임베딩
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        
        # 포지셔널 인코딩
        self.src_pos_encoder = PositionalEncoding(d_model, max_len=src_seq_len)
        self.tgt_pos_encoder = PositionalEncoding(d_model, max_len=tgt_seq_len)
        
        # 트랜스포머 모델, [batch, seq_len, d_model]
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 파라미터 초기화
        self._init_parameters()
        
        # Save hyperparameters for reference
        self.d_model = d_model
        self.vocab_size = vocab_size
        
    def _init_parameters(self):
        """Initialize parameters using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        Args:
            src: 입력 텐서 [batch_size, src_seq_len]
            tgt: 목표 텐서 [batch_size, tgt_seq_len]
            src_mask: 입력 텐서에서 self-attention에 사용할 마스크
            tgt_mask: 목표 텐서에서 self-attention에 사용할 마스크 (usually causal mask)
            memory_mask: encoder-decoder attention에서 사용할 메모리 마스크
            src_key_padding_mask: Source key padding mask [batch_size, src_seq_len]
            tgt_key_padding_mask: Target key padding mask [batch_size, tgt_seq_len]
            memory_key_padding_mask: Memory key padding mask
        """
        # src: [32, 256]
        # tgt: [32, 63]
        # Create embeddings with separate embedding layers
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) # [32, 256, 512]
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model) # [32, 63, 512]
        
        # Add positional encoding with separate encoders
        src_emb = self.src_pos_encoder(src_emb)
        tgt_emb = self.tgt_pos_encoder(tgt_emb)
        
        # Pass through transformer
        output = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        ) # [32, 63, 512]
        
        # Pass through output layer
        return self.output_layer(output) # [32, 63, 192]
    
    def generate(self, src, max_len=100, temperature=1.0, top_k=0, top_p=0.0, 
                 bos_token=188, eos_token=189, pad_token=0):
        """
        Generate a sequence using the trained model.
        
        Args:
            src: Source sequence tensor [batch_size, src_seq_len]
            max_len: Maximum length of the generated sequence
            temperature: Temperature for sampling. Higher values produce more diverse samples
            top_k: Sample from the top k most likely tokens
            top_p: Sample from the smallest set of tokens whose cumulative probability exceeds p
            bos_token: Beginning of sequence token
            eos_token: End of sequence token
            pad_token: Padding token
            
        Returns:
            Generated sequence tensor [batch_size, seq_len]
        """
        device = src.device
        batch_size = src.size(0)
        # src [batch_size, 256]
        
        # Create source mask
        src_key_padding_mask = (src == pad_token)
        
        # Encode source sequence
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model) # [batch_size, 256, 512]
        src_emb = self.src_pos_encoder(src_emb) # [batch_size, 256, 512]
        memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask) # [batch_size, 256, 512]        
        
        # Start with beginning of sequence token
        cur_tokens = torch.full((batch_size, 1), bos_token, dtype=torch.long, device=device) # [batch_size, 1]
        
        # 최대 생성 길이를 target 시퀀스 길이로 제한
        max_len = min(max_len, self.tgt_seq_len) # 64
        
        for i in range(max_len - 1):
            # Don't update tokens that have already generated EOS
            if (cur_tokens == eos_token).any(dim=1).all():
                break
                
            # Create target mask (to prevent attending to future tokens)
            tgt_mask = self.transformer.generate_square_subsequent_mask(cur_tokens.size(1)).to(device) # [1, 1]
            
            # Create target padding mask
            tgt_key_padding_mask = (cur_tokens == pad_token)
            
            # Decode current sequence - 타겟 임베딩 사용
            tgt_emb = self.tgt_embedding(cur_tokens) * math.sqrt(self.d_model) # [batch_size, 1, 512]
            tgt_emb = self.tgt_pos_encoder(tgt_emb) # [batch_size, 1, 512]
            
            # Get transformer decoder output
            decoder_output = self.transformer.decoder(
                tgt_emb, 
                memory, 
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            ) # [batch_size, 1, 512]
            
            # Project to vocabulary space
            logits = self.output_layer(decoder_output[:, -1]) # [batch_size, vocab_size]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-k sampling if specified
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus) sampling if specified
            if top_p > 0.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Concatenate new token to current sequence
            cur_tokens = torch.cat([cur_tokens, next_token], dim=1)
        return cur_tokens

def create_masks(src, tgt, pad_token=0):
    """
    Create masks for transformer training.
    
    Args:
        src: Source tensor [batch_size, src_seq_len]
        tgt: Target tensor [batch_size, tgt_seq_len]
        pad_token: Padding token id
        
    Returns:
        src_mask: Source mask for self-attention
        tgt_mask: Target mask for self-attention (causal mask)
        src_padding_mask: Source padding mask
        tgt_padding_mask: Target padding mask
    """
    device = src.device
    
    # Source mask is None as we want to attend to all positions
    src_mask = None
    
    # Target mask is a causal mask to prevent attending to future positions
    tgt_seq_len = tgt.size(1)
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_seq_len).to(device)
    
    # Padding masks
    src_padding_mask = (src == pad_token)
    tgt_padding_mask = (tgt == pad_token)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

class MIDITransformerTrainer:
    def __init__(self, model, optimizer, scheduler=None, pad_token=0, bos_token=188, eos_token=189):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_token)
    '''
    학습 1회 진행
    epoch는 외부에서 선언할 것
    '''
    def train_epoch(self, dataloader, device):
        self.model.train() # 학습 모드
        total_loss = 0

        num_batches = len(dataloader)
        print(f'num_batches: {num_batches}')
        hundredth = num_batches // 100 + 1
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src, tgt = src.to(device), tgt.to(device) # 텐서 GPU로 이동
            
            tgt_input = tgt[:, :-1] # 목표 입력은 마지막 항목 제외: 트랜스포머에서 디코더로 들어감
            tgt_output = tgt[:, 1:] # 목표 출력은 첫 항목 제외: 트랜스포머에서 출력되어야 하는 것들
            
            # 트랜스포머에서 사용할 마스크 생성
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt_input, self.pad_token)

            output = self.model(
                src, 
                tgt_input, 
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_padding_mask,
                tgt_key_padding_mask=tgt_padding_mask
            ) # 예측
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1)) # 손실값 계산
            self.optimizer.zero_grad() # 기울기 초기화
            loss.backward() # 역전파
            self.optimizer.step() # 가중치 업데이트
            total_loss += loss.item() # 총 손실값 계산
            
            if batch_idx > 0 and batch_idx % hundredth == 0: # 매 10%마다 배치, 손실값 출력
                print(f"Batch: {batch_idx} ({int(batch_idx / num_batches * 100)}%), Loss: {loss.item():.4f}")
                
        # 학습률 업데이트
        if self.scheduler is not None:
            self.scheduler.step()
            
        return total_loss / num_batches
    '''
    평가 1회 진행
    epoch는 외부에서 선언할 것
    '''
    def evaluate(self, dataloader, device):
        self.model.eval() # 평가 모드
        total_loss = 0
        
        with torch.no_grad(): # 역전파 비활성화
            for src, tgt in dataloader:
                src, tgt = src.to(device), tgt.to(device) # 텐서 GPU로 이동

                tgt_input = tgt[:, :-1] # 목표 입력은 마지막 항목 제외: 트랜스포머에서 디코더로 들어감
                tgt_output = tgt[:, 1:] # 목표 출력은 첫 항목 제외: 트랜스포머에서 출력되어야 하는 것들
                
                # 트랜스포머에서 사용할 마스크 생성
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_masks(src, tgt_input, self.pad_token)
                

                output = self.model(
                    src, 
                    tgt_input, 
                    src_mask=src_mask,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_padding_mask,
                    tgt_key_padding_mask=tgt_padding_mask
                ) # 예측
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1)) # 손실값 계산
                total_loss += loss.item() # 총 손실값 계산
                
        return total_loss / len(dataloader)
    '''
    체크포인트 저장
    내용: epoch, 모델, optimizer, scheduler, loss
    '''
    def save_checkpoint(self, path, epoch, valid_loss):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'valid_loss': valid_loss,
        }, path)
    '''
    체크포인트 불러오기
    내용: 모델, optimizer, scheduler
    리턴: epoch, loss
    '''
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['valid_loss']