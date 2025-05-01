# midi_tokenizer.py
from miditok import REMI
from miditoolkit import MidiFile
from pathlib import Path
import os
import glob

import torch
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader



class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_head=4, num_layers=4, dim_ff=1024, max_seq_len=2048):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, max_seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=dim_ff, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq_len)
        seq_len = x.size(1)
        x = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        x = self.transformer(x)
        return self.out(x)  # (batch, seq_len, vocab_size)

class MIDITokenDataset(Dataset):
    def __init__(self, token_seqs, seq_len):
        self.data = []
        for seq in token_seqs:
            for i in range(0, len(seq) - seq_len):
                self.data.append((seq[i:i+seq_len], seq[i+1:i+seq_len+1]))
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x), torch.tensor(y)

def doIt():
    config_dir = Path("miditok_config")
    config_dir.mkdir(exist_ok=True)

    midi_folder = os.path.join("midi","train")
    save_folder = Path("tokenized_ids")
    save_folder.mkdir(exist_ok=True)
    print("파일 경로 불러오는 중")

    # 토크나이저 초기화 (옵션은 기본 설정 사용 가능)
    tokenizer = REMI()
    tokenizer.save_params(config_dir)

    # MIDI 파일 경로
    print("파일 경로에서 파일 읽어오는 중")
    midi_files = glob.glob(os.path.join(midi_folder, '**/*.midi'), recursive=True)

    for path in midi_files:
        try:
            midi = MidiFile(path)
            tokens = tokenizer(midi)
            filename = Path(path).stem + ".json"
            tokens.save(save_folder / filename)  # JSON 파일로 저장
            print(f"{filename} 토큰화 성공")
        except Exception as e:
            print(f"토큰화 실패: {path} | {e}")

    json_folder = Path("tokenized_ids")
    seq_len = 128
    token_seqs = []

    for f in json_folder.glob("*.json"):
        try:
            tokens = tokenizer.load(f)
            token_seqs.append(tokens.ids)
        except:
            print(f"불러오기 실패: {f}")

    dataset = MIDITokenDataset(token_seqs, seq_len)

    features, labels = [], []
    for x, y in dataset:
        features.append(x)
        labels.append(y)

    features = torch.stack(features)
    labels = torch.stack(labels)

    torch.save((features, labels), "dataset.pt")


def main():
    token_seqs = [[1, 2, 3, 4, 5, 6, 7, 8, 9]] * 100  # REMI 토큰으로 대체
    vocab_size = 512
    seq_len = 32
    batch_size = 16

    dataset = MIDITokenDataset(token_seqs, seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicTransformer(vocab_size=vocab_size).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    doIt()