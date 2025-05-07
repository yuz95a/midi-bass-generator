TOKEN_MAP = {
    'PAD': 0, # 패딩용 토큰
    'Position': list(range(1, 1 + 16)),
    'Drumhit': 17,
    'Pitch': list(range(18, 18 + 128)),
    'Velocity': list(range(146, 146 + 32)),
    'Duration': list(range(178, 178 + 9)),
    'MASK': 187, # 마스킹
    'BOS': 188, # 시작
    'EOS': 189, # 끝
    'BAR': 190, # 마디 구분
    'DUMMY': 191 # vocab size를 8의 배수로 맞추기 위한 더미
}

TICKS_PER_BEAT = 480

DURATION = [
    TICKS_PER_BEAT * 4,      # 온음표
    TICKS_PER_BEAT * 2,      # 2분음표
    TICKS_PER_BEAT,          # 4분음표
    TICKS_PER_BEAT // 2,     # 8분음표 
    TICKS_PER_BEAT // 4,     # 16분음표
    TICKS_PER_BEAT // 8,     # 32분음표
    TICKS_PER_BEAT * 3 // 2, # 점4분음표
    TICKS_PER_BEAT * 3 // 4, # 점8분음표
    TICKS_PER_BEAT * 3 // 8  # 점16분음표
]
