def convert_to_grouped_events(events):
    if not events:
        return []
    
    grouped_events = []
    current_time = events[0]['time']
    current_group = []
    
    for event in events:
        if event['time'] == current_time:
            current_group.append(event)
        else:
            grouped_events.append((current_time, current_group))
            current_time = event['time']
            current_group = [event]
    
    # 마지막 그룹 추가
    if current_group:
        grouped_events.append((current_time, current_group))
    
    return grouped_events