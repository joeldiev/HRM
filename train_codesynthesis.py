"""
Training script for CodeSynthesis HRM model.
Trains HRM on code analysis tasks (assumption detection, challenge generation, quality assessment).
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
from tqdm import tqdm
import json
import argparse
from pathlib import Path

# Add HRM modules to path
sys.path.append('/Users/joel/GitHub/HRM')

from models.codesynthesis_adapter import (
    CodeSynthesisPuzzleDataset,
    CodeSynthesisHRM,
    CodeSynthesisHRMConfig,
    CodeSynthesisTokenizer
)
from models.losses import ACTLossHead, softmax_cross_entropy


class CodeSynthesisTrainer:
    """Trainer for CodeSynthesis HRM model."""
    
    def __init__(
        self,
        model: CodeSynthesisHRM,
        train_dataset: CodeSynthesisPuzzleDataset,
        test_dataset: CodeSynthesisPuzzleDataset,
        config: dict
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config
        
        # Set device (MPS for Apple Silicon, CUDA if available, else CPU)
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS (Apple Silicon GPU)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using CUDA GPU")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        
        self.model.to(self.device)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.get("batch_size", 8),
            shuffle=False,  # Set to False for compatibility
            num_workers=0  # Set to 0 for MPS compatibility
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.get("batch_size", 8),
            shuffle=False,
            num_workers=0
        )
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.get("lr", 1e-4),
            weight_decay=config.get("weight_decay", 0.01)
        )
        
        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get("epochs", 50),
            eta_min=config.get("lr_min", 1e-6)
        )
        
        # Setup loss function
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=train_dataset.vocab["<PAD>"])
        self.quality_criterion = torch.nn.MSELoss()
        
        # Initialize wandb if enabled
        if config.get("use_wandb", True):
            wandb.init(
                project="codesynthesis-hrm",
                config=config,
                name=f"hrm-{config.get('run_name', 'default')}"
            )
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0
        total_samples = 0
        task_losses = {"assumption_detection": 0, "challenge_generation": 0, "quality_assessment": 0}
        task_counts = {"assumption_detection": 0, "challenge_generation": 0, "quality_assessment": 0}
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for batch in pbar:
            self.optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids)
            
            # Calculate loss based on task type
            batch_loss = 0
            batch_size = input_ids.size(0)
            
            for i in range(batch_size):
                metadata = batch["metadata"][i]
                task_type = self._infer_task_type(metadata)
                
                if task_type == "quality_assessment":
                    # Use quality head for quality assessment
                    quality_output = self.model.quality_head(outputs.last_hidden_state[i])
                    target_quality = self._extract_quality_score(labels[i])
                    loss = self.quality_criterion(quality_output.squeeze(), target_quality)
                else:
                    # Use standard language modeling loss
                    logits = outputs.logits[i]
                    target = labels[i]
                    loss = self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                
                batch_loss += loss
                task_losses[task_type] += loss.item()
                task_counts[task_type] += 1
            
            # Backward pass
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += batch_loss.item()
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{batch_loss.item():.4f}",
                "avg_loss": f"{total_loss / len(pbar):.4f}"
            })
        
        # Calculate average losses
        avg_loss = total_loss / len(self.train_loader)
        task_avg_losses = {
            task: task_losses[task] / max(task_counts[task], 1)
            for task in task_losses
        }
        
        return {
            "train_loss": avg_loss,
            "train_task_losses": task_avg_losses,
            "train_samples": total_samples
        }
    
    def evaluate(self) -> dict:
        """Evaluate the model."""
        self.model.eval()
        
        total_loss = 0
        total_samples = 0
        task_accuracies = {"assumption_detection": [], "challenge_generation": [], "quality_assessment": []}
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc="Evaluating")
            
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(input_ids)
                
                batch_loss = 0
                batch_size = input_ids.size(0)
                
                for i in range(batch_size):
                    metadata = batch["metadata"][i]
                    task_type = self._infer_task_type(metadata)
                    
                    if task_type == "quality_assessment":
                        quality_output = self.model.quality_head(outputs.last_hidden_state[i])
                        target_quality = self._extract_quality_score(labels[i])
                        loss = self.quality_criterion(quality_output.squeeze(), target_quality)
                        
                        # Calculate accuracy (within 0.1 of target)
                        accuracy = (abs(quality_output.squeeze() - target_quality) < 0.1).float().item()
                        task_accuracies[task_type].append(accuracy)
                    else:
                        logits = outputs.logits[i]
                        target = labels[i]
                        loss = self.criterion(logits.view(-1, logits.size(-1)), target.view(-1))
                        
                        # Calculate token accuracy
                        predictions = torch.argmax(logits, dim=-1)
                        mask = target != self.train_dataset.vocab["<PAD>"]
                        correct = (predictions == target) & mask
                        accuracy = correct.sum().float() / mask.sum().float()
                        task_accuracies[task_type].append(accuracy.item())
                    
                    batch_loss += loss
                
                total_loss += batch_loss.item() / batch_size
                total_samples += batch_size
        
        # Calculate average metrics
        avg_loss = total_loss / len(self.test_loader)
        avg_accuracies = {
            task: sum(accs) / len(accs) if accs else 0.0
            for task, accs in task_accuracies.items()
        }
        
        return {
            "eval_loss": avg_loss,
            "eval_accuracies": avg_accuracies,
            "eval_samples": total_samples
        }
    
    def train(self, epochs: int):
        """Train the model for specified epochs."""
        
        best_eval_loss = float('inf')
        patience = self.config.get("patience", 10)
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            eval_metrics = self.evaluate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            metrics = {
                "epoch": epoch + 1,
                "lr": self.scheduler.get_last_lr()[0],
                **train_metrics,
                **eval_metrics
            }
            
            if self.config.get("use_wandb", True):
                wandb.log(metrics)
            
            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
            print(f"Eval Accuracies: {eval_metrics['eval_accuracies']}")
            
            # Save checkpoint if best model
            if eval_metrics['eval_loss'] < best_eval_loss:
                best_eval_loss = eval_metrics['eval_loss']
                patience_counter = 0
                self.save_checkpoint(epoch, "best_model.pt", metrics)
                print("✅ New best model saved!")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            # Save regular checkpoint
            if (epoch + 1) % self.config.get("save_every", 10) == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch + 1}.pt", metrics)
    
    def save_checkpoint(self, epoch: int, filename: str, metrics: dict):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.get("checkpoint_dir", "checkpoints/codesynthesis-hrm"))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "vocab": self.train_dataset.vocab
        }
        
        filepath = checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def _infer_task_type(self, metadata: dict) -> str:
        """Infer task type from metadata."""
        # This would be based on how the dataset encodes task types
        # For now, assume it's in metadata or infer from other fields
        if "task_type" in metadata:
            return metadata["task_type"]
        elif "quality_score" in metadata:
            return "quality_assessment"
        elif "assumption_count" in metadata:
            return "assumption_detection"
        else:
            return "challenge_generation"
    
    def _extract_quality_score(self, labels: torch.Tensor) -> torch.Tensor:
        """Extract quality score from labels for quality assessment tasks."""
        # This is a simplified version - in practice, you'd decode the label
        # to extract the quality score from text like "quality_score:0.85"
        return torch.tensor(0.5, device=self.device)  # Placeholder


def main():
    parser = argparse.ArgumentParser(description="Train CodeSynthesis HRM model")
    parser.add_argument("--data_dir", type=str, default="data/codesynthesis-hrm", help="Data directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--run_name", type=str, default="default", help="Run name for wandb")
    
    args = parser.parse_args()
    
    # Load datasets
    train_path = os.path.join(args.data_dir, "codesynthesis_train.json")
    test_path = os.path.join(args.data_dir, "codesynthesis_test.json")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"❌ Dataset files not found!")
        print(f"Expected: {train_path} and {test_path}")
        print("Run dataset/build_codesynthesis_dataset.py first")
        return
    
    print("Loading datasets...")
    train_dataset = CodeSynthesisPuzzleDataset(train_path)
    test_dataset = CodeSynthesisPuzzleDataset(test_path)
    
    # Create model
    print("Creating model...")
    config = CodeSynthesisHRMConfig(
        vocab_size=train_dataset.vocab_size,
        batch_size=args.batch_size
    )
    model = CodeSynthesisHRM(config)
    
    # Create trainer config
    trainer_config = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "use_wandb": not args.no_wandb,
        "run_name": args.run_name,
        "patience": 10,
        "save_every": 5
    }
    
    # Create trainer
    trainer = CodeSynthesisTrainer(model, train_dataset, test_dataset, trainer_config)
    
    print(f"🚀 Starting training for {args.epochs} epochs...")
    print(f"   Device: {trainer.device}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    
    # Train model
    trainer.train(args.epochs)
    
    print("✅ Training completed!")


if __name__ == "__main__":
    main()