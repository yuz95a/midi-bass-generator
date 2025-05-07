import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import os
import pickle
import argparse

from preprocessing.encoding import convert_token_from_midi
from preprocessing.dataset import TokenDataset
from model import TransformerModel, MIDITransformerTrainer
from printinfo import printCUDAinfo

def extract_tokens(input_dir, output_dir):
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith('.midi'):
            continue

        else:
            tokens = convert_token_from_midi(os.path.join(input_dir, filename))
            for i, token in enumerate(tokens):
                with open(os.path.join(output_dir, f'{filename.split(".")[0]}_token_{i}.pkl'), 'wb') as f:
                    pickle.dump(token, f)
                print(f'{filename}_token_{i} 저장 완료')

def main(args):
    # cuda 정보 출력
    printCUDAinfo()

    # device 설정: cuda 사용
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    
    # 데이터셋 로드
    train_dataset = TokenDataset(
        input_dir=os.path.join('token','train'),
        feature_size=256,
        label_size=64
    )

    test_dataset = TokenDataset(
        input_dir=os.path.join('token','test'),
        feature_size=256,
        label_size=64
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # 데이터 로더 출력
    print(f'[DATA INFO] train_loader: {train_loader}')
    print(f'[DATA INFO] test_loader: {test_loader}')

    # 모델 하이퍼 파라미터 설정
    model = TransformerModel(
        vocab_size=192,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        src_seq_len=256,  # feature_size
        tgt_seq_len=64    # label_size
    ).to(device) # GPU로 이동
    
    # 모델 정보 출력
    print(f'[MODEL INFO] {model}')
    print(f'[MODEL INFO] Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    # optimizer 설정
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # scheduler 설정
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # trainer 설정
    trainer = MIDITransformerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pad_token=0,
        bos_token=188,
        eos_token=189
    )
    
    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 학습 전 epoch, best_loss 초기화
    start_epoch = 0
    best_valid_loss = float('inf')

    # 체크포인트부터 다시 시작
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_valid_loss = trainer.load_checkpoint(args.resume)
        print(f'[TRAIN INFO] Resuming from epoch {start_epoch} with validation loss {best_valid_loss:.4f}')
        start_epoch += 1  # 불러온 epoch 값은 이미 진행한 것까지의 값, 이후는 그 다음부터 진행
    
    for epoch in range(start_epoch, args.epochs):
        print(f'[TRAIN INFO] Epoch {epoch + 1}/{args.epochs}')
        
        # 학습 진행
        train_loss = trainer.train_epoch(train_loader, device)
        print(f'[TRAIN INFO] Train loss: {train_loss:.4f}')
        
        # 평가 진행
        valid_loss = trainer.evaluate(test_loader, device)
        print(f'[TRAIN INFO] Validation loss: {valid_loss:.4f}')
        
        # 체크포인트 저장
        trainer.save_checkpoint(
            os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt'),
            epoch + 1,
            valid_loss
        )
        print(f'[TRAIN INFO] checkpoint_epoch_{epoch + 1}.pt is saved')
        
        # loss가 가장 적은 모델은 따로 저장
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            trainer.save_checkpoint(
                os.path.join(args.checkpoint_dir, 'best_model.pt'),
                epoch + 1,
                valid_loss
            )
            print(f'[TRAIN INFO] New best model saved with validation loss {valid_loss:.4f}')
    
    print('Training complete!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for MIDI generation")
    
    # 데이터셋 아규먼트
    parser.add_argument("--train_path", type=str, default="train_dataset.pt", help="Path to training dataset")
    parser.add_argument("--test_path", type=str, default="test_dataset.pt", help="Path to test dataset")
    parser.add_argument("--num_workers", type=str, default=8, help="Number of workers")
    
    # 모델 아규먼트
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # 학습 아규먼트
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_step", type=int, default=10, help="Learning rate step size")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Learning rate decay factor")
    
    # 프로그램 실행 아규먼트
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()

    main(args)