def duration_bins(resolution=24):
        max_duration = 8 * resolution  # 최대 2마디 길이
        duration_bins = [0]
        current_tick = 1
        while current_tick < max_duration:
            duration_bins.append(current_tick)
            if current_tick < 16:
                current_tick += 1
            elif current_tick < 32:
                current_tick += 2
            elif current_tick < 64:
                current_tick += 4
            elif current_tick < 128:
                current_tick += 8
            else:
                current_tick += 16
        
        for i in duration_bins:
            print(i, end=' ')

def position_bins(resolution=24):
    position_bins = np.arange(0, resolution, 1)

    for i in position_bins:
            print(i, end=' ')

def tempo_bins(resolution=24):
    tempo_bins = np.arange(30, 210, 5)
    for i in tempo_bins:
        print(i, end=' ')
