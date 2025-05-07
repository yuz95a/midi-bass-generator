from . import split_by_bar
from . import split_by_instrument
from . import remove_metadata
from . import convert_events
from . import convert_grouped_events
from . import convert_remi_events
from . import convert_tokens
from . import token_map

def convert_token_from_midi(midi_path):
    midis = split_by_bar.split_midi_by_bar(midi_path)
    tokens = []
    for midi in midis:
        bass_tracks, drum_tracks, other_tracks = split_by_instrument.split_midi_by_instrument(midi)

        label_events = []
        for bass in bass_tracks:
            label_events.append(convert_structured_remi_events_from_track(bass))
        feature_events = []
        for drum in drum_tracks:
            feature_events.append(convert_structured_remi_events_from_track(drum))
        for other in other_tracks:
            feature_events.append(convert_structured_remi_events_from_track(other))
        
        feature_tokens, label_tokens = convert_tokens_from_structured_remi_events(label_events, feature_events, midi is midis[-1])

        if feature_tokens:
            data = {'feature': feature_tokens, 'label': label_tokens}
            tokens.append(data)

    return tokens

def convert_structured_remi_events_from_track(track):
    remove_metadata.remove_metadata_in_track(track)
    events = convert_events.convert_to_events(track)
    grouped_events = convert_grouped_events.convert_to_grouped_events(events)
    remi_events = convert_remi_events.convert_to_remi_events(grouped_events)
    quantized_remi_events = convert_remi_events.convert_to_quantize_remi_events(remi_events)
    structured_remi_events = convert_remi_events.convert_to_structured_remi_events(quantized_remi_events)

    return structured_remi_events

def convert_tokens_from_structured_remi_events(label_events, feature_events, is_end):
    _feature_events = convert_remi_events.combine_remi_events(*feature_events)
    feature_tokens = convert_tokens.convert_to_token(_feature_events)

    _label_events = convert_remi_events.combine_remi_events(*label_events)
    label_tokens = convert_tokens.convert_to_token(_label_events)

    if feature_tokens and label_tokens:
        if is_end:
            feature_tokens.append(token_map.TOKEN_MAP['EOS'])
            label_tokens.append(token_map.TOKEN_MAP['EOS'])
    
        return feature_tokens, label_tokens
    else:
        return None, None
