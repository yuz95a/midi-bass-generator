import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import argparse

from preprocessing.token_dataset import load_datasets, create_dataloaders, TokenDataset
from model.midi_transformer import TransformerModel, MIDITransformerTrainer

from printing.print_cuda_info import printCUDAinfo
from printing.print_midi import print_instruments

def main(args):
    printCUDAinfo()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    # print(f"Using device: {device}")

    # Load datasets
    train_dataset, test_dataset = load_datasets()
    train_loader, test_loader = create_dataloaders(train_dataset, test_dataset)
    
    # # Create dataloaders
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, 
    #     batch_size=args.batch_size, 
    #     shuffle=True
    # )
    
    # test_loader = torch.utils.data.DataLoader(
    #     test_data, 
    #     batch_size=args.batch_size, 
    #     shuffle=False
    # )
    
    # Vocabulary size (from your token conversion function)
    # PAD, Position (16), Drumhit, Pitch (128), Velocity (32), Duration (9), MASK, BOS, EOS, BAR, DUMMY
    vocab_size = 192  # Based on your tokenization scheme
    
    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        src_seq_len=256,  # feature_size
        tgt_seq_len=64    # label_size
    ).to(device)
    
    # Print model architecture
    print(model)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create learning rate scheduler
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    
    # Create trainer
    trainer = MIDITransformerTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        pad_token=0,
        bos_token=188,
        eos_token=189
    )
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_valid_loss = float('inf')
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        start_epoch, best_valid_loss = trainer.load_checkpoint(args.resume)
        print(f"Resuming from epoch {start_epoch} with validation loss {best_valid_loss:.4f}")
        start_epoch += 1  # Start from the next epoch
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, device)
        print(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        valid_loss = trainer.evaluate(test_loader, device)
        print(f"Validation loss: {valid_loss:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(
            os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"),
            epoch + 1,
            valid_loss
        )
        
        # Save best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            trainer.save_checkpoint(
                os.path.join(args.checkpoint_dir, "best_model.pt"),
                epoch + 1,
                valid_loss
            )
            print(f"New best model saved with validation loss {valid_loss:.4f}")
    
    print("Training complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Transformer model for MIDI generation")
    
    # Data parameters
    parser.add_argument("--train_path", type=str, default="train_dataset.pt", help="Path to training dataset")
    parser.add_argument("--test_path", type=str, default="test_dataset.pt", help="Path to test dataset")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers")
    parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of feedforward network")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--lr_step", type=int, default=10, help="Learning rate step size")
    parser.add_argument("--lr_gamma", type=float, default=0.9, help="Learning rate decay factor")
    parser.add_argument("--use_cuda", action="store_true", help="Use CUDA")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    
    args = parser.parse_args()
    
    main(args)