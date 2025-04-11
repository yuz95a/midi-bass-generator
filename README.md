# MIDI 베이스 트랙 생성기

이 리포지토리는 트랜스포머 기반 딥러닝 모델을 사용하여 MIDI 파일의 다른 악기 트랙들로부터 베이스 트랙을 자동으로 생성하는 프로젝트입니다.

## 데이터셋

이 프로젝트에서 사용된 MIDI 데이터셋은 다음 링크에서 가져왔습니다:
- [XMIDI Dataset](https://github.com/xmusic-project/XMIDI_Dataset?tab=readme-ov-file)

## 프로젝트 개요

이 프로젝트는 트랜스포머 인코더-디코더 아키텍처를 사용하여 MIDI 파일에서 다른 악기 트랙들을 분석하고, 이에 어울리는 베이스 트랙을 자동으로 생성합니다. 주요 기능은 다음과 같습니다:

- MIDI 파일 로드 및 피아노 롤 변환
- 베이스 트랙과 다른 트랙 분리
- 트랜스포머 기반 시퀀스-투-시퀀스 모델 학습
- 자기회귀(auto-regressive) 방식의 베이스 트랙 생성
- 생성된 베이스 트랙을 MIDI 파일로 저장

## 사용 방법

### 설치 요구사항

```bash
pip install torch pretty_midi safetensors numpy
```

### 모델 학습

```bash
python train.py
```

### 베이스 트랙 생성

```bash
python predict.py --input input_midi_file.mid --output output_with_bass.mid --model models/model1.safetensors
```

## 프로젝트 구조

- `train.py`: 모델 학습 스크립트
- `transformer.py`: 트랜스포머 모델 구현
- `midi_bass_dataset.py`: MIDI 데이터셋 처리
- `predict.py`: 학습된 모델을 사용한 베이스 트랙 생성
- `models/`: 학습된 모델 가중치 저장 디렉토리

## 주요 기술

- PyTorch를 이용한 딥러닝 모델 구현
- Transformer 인코더-디코더 아키텍처
- MIDI 파일 처리 (pretty_midi 라이브러리)
- Safetensors를 이용한 모델 가중치 저장 및 로드
- 자기회귀(auto-regressive) 생성 방식

## 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다.