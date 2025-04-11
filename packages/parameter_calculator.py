import torch
from transformer import MIDIBassGenerator

def calculate_transformer_params(input_dim, hidden_dim, num_layers, nhead, output_dim=None):
    """트랜스포머 모델의 파라미터 수를 이론적으로 계산"""
    if output_dim is None:
        output_dim = input_dim
    
    # 입출력 프로젝션
    input_projection = input_dim * hidden_dim + hidden_dim
    output_projection = hidden_dim * output_dim + output_dim
    
    # 인코더 레이어 파라미터 (1개)
    # 멀티헤드 어텐션
    mha_params = 4 * hidden_dim * hidden_dim  # Q, K, V 변환 + 출력 프로젝션
    
    # 피드포워드 네트워크
    ff_dim = hidden_dim * 4  # 표준 트랜스포머 비율
    ff_params = hidden_dim * ff_dim + ff_dim + ff_dim * hidden_dim + hidden_dim
    
    # 레이어 노멀라이제이션
    ln_params = 4 * hidden_dim  # 2개의 레이어 노멀라이제이션(각각 gamma, beta)
    
    encoder_layer_params = mha_params + ff_params + ln_params
    
    # 디코더 레이어 파라미터 (1개)
    # 디코더는 셀프 어텐션과 인코더-디코더 어텐션이 있음
    decoder_layer_params = 2 * mha_params + ff_params + 6 * hidden_dim
    
    # 총 파라미터 수
    total_params = input_projection + output_projection + \
                   (encoder_layer_params * num_layers) + \
                   (decoder_layer_params * num_layers)
    
    return total_params

def get_actual_params(model):
    """PyTorch 모델의 실제 파라미터 수를 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # 설정값들
    input_dim = 128
    output_dim = 128
    
    # 테스트할 구성들
    configurations = [
        # (이름, hidden_dim, num_layers, nhead)
        ("소형 모델 (5.6M)", 256, 3, 8),
        ("중형 모델 (25M)", 512, 6, 8),
        ("목표 모델 (50M)", 768, 7, 12)
    ]
    
    print(f"{'모델 이름':<15} {'Hidden':<8} {'Layers':<8} {'Heads':<8} {'이론적 파라미터':<20} {'실제 파라미터':<20} {'차이':<10}")
    print("-" * 90)
    
    for name, hidden_dim, num_layers, nhead in configurations:
        # 이론적 계산
        theoretical_params = calculate_transformer_params(
            input_dim, hidden_dim, num_layers, nhead, output_dim
        )
        
        # 실제 모델 생성 및 파라미터 수 계산
        model = MIDIBassGenerator(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            nhead=nhead,
            output_dim=output_dim
        )
        actual_params = get_actual_params(model)
        
        # 차이 계산
        diff_percent = abs(theoretical_params - actual_params) / actual_params * 100
        
        print(f"{name:<15} {hidden_dim:<8} {num_layers:<8} {nhead:<8} {theoretical_params:,} {actual_params:,} {diff_percent:.2f}%")
    
    # 목표 50M 파라미터 모델에 대한 상세 분석
    target_model = MIDIBassGenerator(
        input_dim=input_dim,
        hidden_dim=768,
        num_layers=7,
        nhead=12,
        output_dim=output_dim
    )
    
    target_params = get_actual_params(target_model)
    print("\n50M 목표 모델 분석:")
    print(f"총 파라미터 수: {target_params:,}")
    
    # 메모리 사용량 계산
    fp32_memory = target_params * 4 / (1024**2)  # MB
    fp16_memory = target_params * 2 / (1024**2)  # MB
    print(f"FP32 메모리 사용량: {fp32_memory:.2f} MB")
    print(f"FP16 메모리 사용량: {fp16_memory:.2f} MB")