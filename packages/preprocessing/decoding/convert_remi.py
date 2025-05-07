from . import token_map

def convert_to_remi_events(tokens, token_map=token_map.TOKEN_MAP, durations=token_map.DURATION):
    remi_events = []

    if not tokens:
        return remi_events

    for token in tokens:
        if token in token_map['Position']:
            position_event = {'type': 'Position', 'value': 1920 * (token - token_map['Position'][0]) / 16}
            remi_events.append(position_event)
        elif token == 20:
            drum_event = {'type': 'Drumhit', 'value': True}
            remi_events.append(drum_event)
        elif token in token_map['Pitch']:
            pitch_event = {'type': 'Pitch', 'value': token - token_map['Pitch'][0]}
            remi_events.append(pitch_event)
        elif token in token_map['Velocity']:
            velocity_token = {'type': 'Velocity', 'value': (token - token_map['Velocity'][0]) * 4}
            remi_events.append(velocity_token)
        elif token in token_map['Duration']:
            duration_event = {'type': 'Duration', 'value': durations[token - token_map['Duration'][0]]}
            remi_events.append(duration_event)

    return remi_events