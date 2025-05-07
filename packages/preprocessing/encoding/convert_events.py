def convert_to_events(track):
    events = []
    absolute_time = 0

    for msg in track:
        absolute_time += msg.time

        if msg.type == 'note_on':
            event_type = 'note_on' if msg.velocity > 0 else 'note_off'
            events.append({
                'type': event_type,
                'time': absolute_time,
                'note': msg.note,
                'velocity': msg.velocity,
                'channel': msg.channel
            })
    events.sort(key=lambda x: x['time'])
    return events