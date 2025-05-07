from . import token_map

def convert_to_token(remi_events, eos=False, token_map=token_map.TOKEN_MAP):

    tokens = []

    if not remi_events:
        return tokens
    
    tokens.append(token_map['BOS'])
    for event in remi_events:
        if event['type'] == 'Position':
            position_token = token_map['Position'][event['value']]
            tokens.append(position_token)
        elif event['type'] == 'Drumhit':
            drum_token = token_map['Drumhit']
            tokens.append(drum_token)
        elif event['type'] == 'Pitch':
            pitch_token = token_map['Pitch'][event['value']]
            tokens.append(pitch_token)
        elif event['type'] == 'Velocity':
            # 0-127 벨로시티를 0-31 범위로 양자화
            velocity_level = min(31, event['value'] // 4)
            velocity_token = token_map['Velocity'][velocity_level]
            tokens.append(velocity_token)
        elif event['type'] == 'Duration':
            duration_token = token_map['Duration'][event['value']]
            tokens.append(duration_token)
    
    tokens.append(token_map['BAR'])

    if eos:
        tokens.append(token_map['EOS'])

    return tokens