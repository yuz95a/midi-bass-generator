import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MIDIBassGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MIDIBassGenerator, self).__init__()
        
        # 인코더 (다른 트랙들을 처리)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=8,
                batch_first=True
            ), 
            num_layers=num_layers
        )
        
        # 디코더 (베이스 트랙 생성)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim, 
                nhead=8,
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # 입출력 프로젝션
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, output_dim)
        
        # 위치 인코딩 추가
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout=0.1, max_len=1000)
        
    def forward(self, src, tgt):
        # src: [batch_size, seq_len, input_dim]
        # tgt: [batch_size, seq_len, input_dim]
        
        # 입력 투영
        src = self.input_projection(src)
        tgt = self.input_projection(tgt)
        
        # 위치 인코딩 적용 (차원 순서 변경 필요)
        src = src.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        tgt = tgt.transpose(0, 1)  # [seq_len, batch_size, hidden_dim]
        tgt = self.pos_encoder(tgt)
        tgt = tgt.transpose(0, 1)  # [batch_size, seq_len, hidden_dim]
        
        # 인코더 통과 (batch_first=True이므로 permute 불필요)
        memory = self.encoder(src)
        
        # 마스크 생성 (자기회귀 생성을 위한 look-ahead 마스크)
        tgt_mask = generate_square_subsequent_mask(tgt.size(1)).to(tgt.device)
        
        # 디코더 통과 (batch_first=True이므로 permute 불필요)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # 출력 투영
        output = self.output_projection(output)
        
        return output

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