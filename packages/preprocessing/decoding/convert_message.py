def convert_to_midi_events(remi_events):
    positions = []
    pitches = []
    velocities = []
    durations = []

    for events in remi_events:
        if events['type'] == 'Position':
            positions.append(events['value'])
        elif events['type'] == 'Pitch':
            th = events['value']
            while th > 62:
                th -= 12
            pitches.append(th)
        elif events['type'] == 'Velocity':
            velocities.append(events['value'])
        elif events['type'] == 'Duration':
            durations.append(events['value'])

    size = min(len(positions), len(pitches), len(velocities), len(durations))

    position = 0
    events = []
    for i in range(size):
        if position != 0 and position == positions[i]:
            continue
        if position != positions[i]:
            position = positions[i]
            
        events.append({
            'time': positions[i],
            'type':'note_on',
            'pitch': pitches[i], # Pitch, bass 트랙만 midi로 바꾸는 것으로 가정
            'velocity': velocities[i]
        })
        events.append({
            'time': positions[i] + durations[i],
            'type':'note_off',
            'pitch': pitches[i], # Pitch, bass 트랙만 midi로 바꾸는 것으로 가정
            'velocity': velocities[i]
        })
    events.sort(key=lambda x: x['time'])

    midi_events = []
    prev_time = 0

    for e in events:
        delta_time = e['time'] - prev_time
        midi_events.append({
            'delta': int(delta_time),
            'type': e['type'],
            'pitch': e['pitch'],
            'velocity': e['velocity']
        })
        prev_time = e['time']

    return midi_events
