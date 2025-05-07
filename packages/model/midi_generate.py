import torch
import argparse
import os
import pickle
from mido import MidiFile, MidiTrack, Message, merge_tracks
from transformer_model import TransformerModel, MIDITransformerTrainer

TOKEN_MAP = {
    0: {'type': 'PAD', 'value': 0},  # 패딩용 토큰
    
    # Position tokens (1-16)
    **{i: {'type': 'Position', 'value': i-1} for i in range(1, 17)},
    
    # Drumhit token
    17: {'type': 'Drumhit', 'value': True},
    
    # Pitch tokens (18-145)
    **{18+i: {'type': 'Pitch', 'value': i} for i in range(128)},
    
    # Velocity tokens (146-177)
    **{146+i: {'type': 'Velocity', 'value': i*4} for i in range(32)},
    
    # Duration tokens (178-186)
    **{178+i: {'type': 'Duration', 'value': i} for i in range(9)},
    
    # Special tokens
    187: {'type': 'MASK', 'value': 0},
    188: {'type': 'BOS', 'value': 0},
    189: {'type': 'EOS', 'value': 0},
    190: {'type': 'BAR', 'value': 0},
    191: {'type': 'DUMMY', 'value': 0}
}

# Duration map (from quantized durations to actual tick values)
DURATION_MAP = [
    480 * 4,      # 온음표
    480 * 2,      # 2분음표
    480,          # 4분음표
    480 // 2,     # 8분음표 
    480 // 4,     # 16분음표
    480 // 8,     # 32분음표
    480 * 3 // 2, # 점4분음표
    480 * 3 // 4, # 점8분음표
    480 * 3 // 8  # 점16분음표
]

def tokens_to_remi_events(tokens):
    """Convert tokens back to REMI events"""
    events = []
    for token in tokens:
        if token == 0 or token == 187 or token == 188 or token == 189 or token == 190 or token == 191:
            continue  # Skip special tokens
        
        token_info = TOKEN_MAP.get(token.item())
        if token_info:
            events.append(token_info)
    
    return events

def remi_events_to_midi(remi_events, output_file):
    """Convert REMI events back to MIDI file"""
    midi = MidiFile(ticks_per_beat=480)
    
    # Add a metronome track with tempo
    meta_track = MidiTrack()
    midi.tracks.append(meta_track)
    meta_track.append(Message('set_tempo', tempo=500000, time=0))  # 120 BPM
    meta_track.append(Message('time_signature', numerator=4, denominator=4, time=0))
    
    # Add a track for our generated music
    track = MidiTrack()
    midi.tracks.append(track)
    
    # Set instrument (program_change)
    track.append(Message('program_change', program=0, time=0))  # Default to piano
    
    # Process events
    position = 0
    current_position = 0
    notes = {}  # Dictionary to track active notes: (note) -> (start_time, velocity)
    
    for i, event in enumerate(remi_events):
        event_type = event['type']
        
        if event_type == 'Position':
            # Calculate the real time position
            position_value = event['value']
            target_position = int((position_value / 16) * 1920)  # Convert to ticks (16 positions per bar)
            
            # Update current position
            current_position = target_position
            
        elif event_type == 'Pitch':
            note_value = event['value']
            # The velocity will be set when we encounter a Velocity event
            notes[note_value] = {'start_time': current_position, 'velocity': None}
            
        elif event_type == 'Drumhit':
            # For simplicity, we'll map the drum hit to a specific note (e.g., bass drum)
            note_value = 36  # Bass drum in MIDI
            notes[note_value] = {'start_time': current_position, 'velocity': None, 'is_drum': True}
            
        elif event_type == 'Velocity':
            # Set velocity for the most recently added note
            velocity_value = event['value']
            
            # Find the most recent note without a velocity set
            for note_id, note_info in reversed(list(notes.items())):
                if note_info['velocity'] is None:
                    notes[note_id]['velocity'] = velocity_value
                    break
                    
        elif event_type == 'Duration':
            # Add note_on and note_off messages for completed notes
            duration_idx = event['value']
            duration_ticks = DURATION_MAP[duration_idx] if duration_idx < len(DURATION_MAP) else 480  # Default to quarter note
            
            # Find the earliest note with a set velocity to connect this duration to
            for note_id, note_info in list(notes.items()):
                if note_info['velocity'] is not None:
                    start_time = note_info['start_time']
                    velocity = note_info['velocity']
                    
                    # Calculate time delta from the last event
                    if len(track) == 1:  # Only the program_change message
                        time_delta = start_time
                    else:
                        prev_time = 0
                        for msg in reversed(track):
                            if hasattr(msg, 'time'):
                                prev_time += msg.time
                                break
                        time_delta = start_time - prev_time
                    
                    # Add note_on message
                    channel = 9 if note_info.get('is_drum', False) else 0
                    track.append(Message('note_on', note=note_id, velocity=velocity, time=max(0, time_delta), channel=channel))
                    
                    # Add note_off message (velocity=0 for note_off)
                    track.append(Message('note_off', note=note_id, velocity=0, time=duration_ticks, channel=channel))
                    
                    del notes[note_id]  # Remove the note from active notes
                    break
    
    # Add end of track message
    track.append(Message('end_of_track', time=0))
    
    # Save the MIDI file
    midi.save(output_file)
    print(f"MIDI file saved to {output_file}")

def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Load the model
    model_checkpoint = torch.load(args.model_path, map_location=device)
    
    # Vocabulary size
    vocab_size = 192  # Based on your tokenization scheme
    
    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=0.0,  # Set to 0 for generation
        src_seq_len=256,  # feature_size
        tgt_seq_len=64    # label_size
    ).to(device)
    
    # Load model weights
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {args.model_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load seed data
    if args.seed_path:
        with open(args.seed_path, 'rb') as f:
            seed_data = pickle.load(f)
            feature_tokens = torch.tensor(seed_data['feature'], dtype=torch.long).unsqueeze(0).to(device)
    else:
        # Use a simple seed with just BOS token
        feature_tokens = torch.tensor([[188]], dtype=torch.long).to(device)  # BOS token
    
    # Generate new sequence
    generated_tokens = model.generate(
        feature_tokens,
        max_len=args.max_length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Remove batch dimension
    generated_tokens = generated_tokens[0]
    print(f"Generated sequence length: {len(generated_tokens)}")
    