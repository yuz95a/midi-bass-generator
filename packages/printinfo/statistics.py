import statistics

def print_statistics(data, title=None):
    if not data:
        print("리스트가 비어 있습니다.")
        return

    max_val = max(data)
    min_val = min(data)
    avg_val = sum(data) / len(data)
    median_val = statistics.median(data)
    std_dev = statistics.stdev(data) if len(data) > 1 else 0.0

    if title:
        print(f"{'-'*40}{title}{'-'*40}")
    print(f"최댓값: {max_val}")
    print(f"최솟값: {min_val}")
    print(f"평균: {avg_val:.2f}")
    print(f"중앙값: {median_val}")
    print(f"표준편차: {std_dev:.2f}")