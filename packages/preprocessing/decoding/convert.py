from . import convert_remi
from . import convert_message
from . import convert_midi

def convert_midi_from_token(token):
    remi_events = convert_remi.convert_to_remi_events(token)
    midi_events = convert_message.convert_to_midi_events(remi_events)
    midi = convert_midi.convert_to_midi(midi_events)

    return midi
