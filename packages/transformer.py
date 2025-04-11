import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MIDIBassGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, nhead=8, dropout=0.1, output_dim=128):
        super(MIDIBassGenerator, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nhead = nhead
        self.dropout = dropout
        self.output_dim = output_dim
        
        # 인코더 (다른 트랙들을 처리)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,  # 표준 Transformer 비율
            dropout=dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # 디코더 (베이스 트랙 생성)
        decoder_layers = nn.TransformerDecoderLayer(
            d_model=hidden_dim, 
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,  # 표준 Transformer 비율
            dropout=dropout,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        
        # 입출력 프로젝션
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 위치 인코딩 추가
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=dropout, max_len=2000)

        # 가중치 초기화
        self._reset_parameters()

    def _reset_parameters(self):
        """모델 가중치 초기화"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def forward(self, src, tgt):
        # src: [batch_size, seq_len, input_dim]
        # tgt: [batch_size, seq_len, input_dim]
        
        # 입력 투영
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        # 위치 인코딩 적용
        src = src.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        tgt = tgt.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 인코더 통과
        memory = self.encoder(src)
        
        # 마스크 생성 (자기회귀 생성을 위한 look-ahead 마스크)
        tgt_seq_len = tgt.size(1)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len).to(tgt.device)
        
        # 디코더 통과
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 출력 투영
        output = self.output_projection(output)
        
        return output

    def get_parameter_count(self):
        """모델 파라미터 수 계산"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x 형태: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def generate_square_subsequent_mask(sz):
    """마스크 생성 함수: 이후 위치(미래)에 attention이 발생하지 않도록 함"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def count_model_parameters(model):
    """모델의 파라미터 수를 계산하고 컴포넌트별로 출력"""
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 주요 컴포넌트별 파라미터 수
    input_proj_params = sum(p.numel() for p in model.input_projection.parameters() if p.requires_grad)
    output_proj_params = sum(p.numel() for p in model.output_projection.parameters() if p.requires_grad)
    
    encoder_params = sum(p.numel() for name, p in model.named_parameters() 
                         if 'encoder' in name and p.requires_grad)
    decoder_params = sum(p.numel() for name, p in model.named_parameters() 
                         if 'decoder' in name and p.requires_grad)
    
    print(f"모델 파라미터 통계:")
    print(f"- 총 파라미터 수: {total_params:,}")
    print(f"- 입력 프로젝션: {input_proj_params:,} ({input_proj_params/total_params*100:.1f}%)")
    print(f"- 출력 프로젝션: {output_proj_params:,} ({output_proj_params/total_params*100:.1f}%)")
    print(f"- 인코더: {encoder_params:,} ({encoder_params/total_params*100:.1f}%)")
    print(f"- 디코더: {decoder_params:,} ({decoder_params/total_params*100:.1f}%)")
    
    # 메모리 사용량 계산
    fp32_memory = total_params * 4 / (1024**2)  # MB
    fp16_memory = total_params * 2 / (1024**2)  # MB
    print(f"- 모델 가중치 메모리 (FP32): {fp32_memory:.2f} MB")
    print(f"- 모델 가중치 메모리 (FP16): {fp16_memory:.2f} MB")
    
    return total_params


# 테스트 코드
if __name__ == "__main__":
    # 50M 파라미터 규모 모델 생성 및 파라미터 수 확인
    model_50M = MIDIBassGenerator(
        input_dim=128,
        hidden_dim=512,
        num_layers=7,
        nhead=8,
        output_dim=128
    )
    
    params = count_model_parameters(model_50M)
    print(f"목표: 50M 파라미터, 실제: {params:,} 파라미터")