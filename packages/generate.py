import os
import argparse
import torch

from printinfo import printCUDAinfo
from preprocessing.dataset import pad_sequence
from preprocessing.encoding import convert_token_from_midi
from preprocessing.decoding import convert_midi_from_token
from postprocessing import combine_midis
from model import TransformerModel

def main(args):
    # cuda 정보 출력
    printCUDAinfo()

    # device 설정: cuda 사용
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

    # 체크포인트 로드
    model_checkpoint = torch.load(args.model_path, map_location=device)
    
    vocab_size = 192
    
    # 모델 설정
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,
        src_seq_len=256,
        tgt_seq_len=64
    ).to(device)
    
    # 모델 가중치 로드
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()
    
    print(f'Model loaded from {args.model_path}')
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 전처리 진행
    tokens = None
    if args.seed_path:
        tokens = convert_token_from_midi(args.seed_path)
    else:
        raise Exception(f'seed_path 없음')
    
    padded_features = []

    if tokens is None:
        raise Exception(f'midi 파일에서 전처리 실패')

    for data in tokens:
        feature_seq = data['feature']
        padded_seq = pad_sequence(feature_seq, 256)
        padded_features.append(padded_seq)
    feature_tokens = torch.tensor(padded_features)

    # 토큰을 모델에 넣어야 하는데 한 번에 전부 넣으면 batch_size가 커지는 문제 발생
    # batch_size를 지정해서 배치씩 모델에 입력
    batches = []
    batch_size = 32
    for i in range(0, feature_tokens.size(0), batch_size):
        batch = feature_tokens[i:i + batch_size]
        batches.append(batch)

    generated_tokens = []
    for batch in batches:
        feature_tokens = batch.to(device)
        # 모델로 토큰 생성
        generated_tokens.append(model.generate(
            feature_tokens,
            max_len=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        ))
    generated_tokens = torch.cat(generated_tokens, dim=0)
    
    # 후처리 진행 후 midi 파일 반환
    midis = []
    for tokens in generated_tokens:
        midis.append(convert_midi_from_token(tokens.tolist()))

    new_midi = combine_midis(midis)
    new_midi.save(os.path.join(args.output_dir, args.output_filename))
    print(f'{args.output_filename} 생성 완료')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for MIDI generation")

    # 모델 아규먼트
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of feedforward network")

    # 생성 아규먼트
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the generated sequence")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for sampling. Higher values produce more diverse samples")
    parser.add_argument("--top_k", type=int, default=0, help="Sample from the top k most likely tokens")
    parser.add_argument("--top_p", type=float, default=0.0, help="Sample from the smallest set of tokens whose cumulative probability exceeds p")

    # 프로그램 실행 아규먼트
    parser.add_argument("--seed_path", type=str, help="Path to input midi file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save outputs")
    parser.add_argument("--output_filename", type=str, help="Directory to save new midi file")
    parser.add_argument("--model_path", type=str, default="checkpoints/best_model.pt", help="Path to load model")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA")
    
    args = parser.parse_args()

    if args.output_filename is None:
        args.output_filename = os.path.splitext(os.path.basename(args.seed_path))[0] + "_bass.midi"

    main(args)