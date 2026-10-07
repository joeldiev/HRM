"""
HRM adapter for CodeSynthesis puzzle format.
Adapts the generic HRM model to work with code analysis puzzles.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
import json
import os
from pathlib import Path

# Import HRM components
from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config
)
from puzzle_dataset import PuzzleDataset
from models.common import trunc_normal_init_


class CodeSynthesisPuzzleDataset(torch.utils.data.Dataset):
    """Dataset adapter for CodeSynthesis puzzles."""
    
    def __init__(self, data_path: str, max_seq_len: int = 512):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        
        # Load puzzles
        with open(data_path, 'r') as f:
            self.puzzles = json.load(f)
        
        # Build vocabulary from code and outputs
        self.vocab = self._build_vocabulary()
        self.vocab_size = len(self.vocab)
        
        print(f"Loaded {len(self.puzzles)} CodeSynthesis puzzles")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary from puzzle data."""
        
        # Base vocabulary
        vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<START>": 2,
            "<END>": 3,
            "<SEP>": 4,  # Separator between input and output
        }
        
        # Add common programming tokens
        programming_tokens = [
            # Python keywords
            "def", "class", "import", "from", "if", "else", "elif", "for", "while",
            "try", "except", "finally", "with", "as", "return", "yield", "lambda",
            "and", "or", "not", "in", "is", "None", "True", "False",
            
            # Common patterns
            "self", "user_id", "data", "result", "value", "error", "response",
            "request", "database", "cache", "file", "config", "api", "service",
            
            # Task-specific tokens
            "assumption", "challenge", "quality", "issue", "validation", "error_handling",
            "security", "performance", "edge_case", "documentation", "testing",
            
            # Symbols and operators
            "(", ")", "[", "]", "{", "}", "=", "+", "-", "*", "/", "%", ".",
            ",", ":", ";", "?", "!", "&", "|", "^", "~", "<", ">", "\"", "'",
            
            # Quality indicators
            "high", "medium", "low", "good", "bad", "missing", "present", "required"
        ]
        
        # Add programming tokens
        for token in programming_tokens:
            if token not in vocab:
                vocab[token] = len(vocab)
        
        # Extract tokens from puzzles
        all_text = []
        for puzzle in self.puzzles:
            # Add code tokens
            code = puzzle["input"]["code"]
            all_text.extend(self._tokenize_code(code))
            
            # Add output tokens
            for output in puzzle["output"]:
                all_text.extend(self._tokenize_text(output))
            
            # Add context tokens
            context = puzzle["input"]["context"]
            all_text.extend(self._tokenize_text(context))
        
        # Add frequent tokens to vocabulary
        token_counts = {}
        for token in all_text:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        # Add tokens that appear at least 3 times
        for token, count in token_counts.items():
            if count >= 3 and token not in vocab:
                vocab[token] = len(vocab)
        
        return vocab
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code while preserving structure."""
        import re
        
        # Simple tokenization - could be enhanced with AST
        # Split on whitespace and common separators
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [token.lower() for token in tokens if token.strip()]
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize regular text."""
        import re
        
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return [token for token in tokens if token.strip()]
    
    def __len__(self) -> int:
        return len(self.puzzles)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        puzzle = self.puzzles[idx]
        
        # Encode input (code + task + context)
        input_tokens = []
        input_tokens.append("<START>")
        
        # Add task type
        task = puzzle["input"]["task"]
        input_tokens.extend(self._tokenize_text(task))
        input_tokens.append("<SEP>")
        
        # Add code
        code = puzzle["input"]["code"]
        input_tokens.extend(self._tokenize_code(code))
        input_tokens.append("<SEP>")
        
        # Add context
        context = puzzle["input"]["context"]
        input_tokens.extend(self._tokenize_text(context))
        
        # Encode output
        output_tokens = []
        output_tokens.append("<SEP>")
        for output in puzzle["output"]:
            output_tokens.extend(self._tokenize_text(output))
            output_tokens.append("<SEP>")
        output_tokens.append("<END>")
        
        # Combine input and output
        all_tokens = input_tokens + output_tokens
        
        # Convert to token IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.vocab.get(token, self.vocab["<UNK>"]))
        
        # Pad or truncate
        if len(token_ids) > self.max_seq_len:
            token_ids = token_ids[:self.max_seq_len]
        else:
            token_ids.extend([self.vocab["<PAD>"]] * (self.max_seq_len - len(token_ids)))
        
        # Create input/output split
        sep_idx = token_ids.index(self.vocab["<SEP>"]) if self.vocab["<SEP>"] in token_ids else len(input_tokens)
        
        input_ids = token_ids[:sep_idx] + [self.vocab["<PAD>"]] * (len(token_ids) - sep_idx)
        output_ids = [self.vocab["<PAD>"]] * sep_idx + token_ids[sep_idx:]
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(output_ids, dtype=torch.long),
            "metadata": puzzle["metadata"]
        }


class CodeSynthesisHRMConfig(HierarchicalReasoningModel_ACTV1Config):
    """Configuration for CodeSynthesis HRM model."""
    
    def __init__(self, vocab_size: int, **kwargs):
        # CodeSynthesis-specific defaults
        defaults = {
            "batch_size": 16,
            "seq_len": 512,
            "puzzle_emb_ndim": 0,
            "num_puzzle_identifiers": 1,
            "vocab_size": vocab_size,
            
            # Hierarchical reasoning config
            "H_cycles": 4,  # High-level reasoning cycles
            "L_cycles": 8,  # Low-level reasoning cycles
            "H_layers": 6,  # High-level transformer layers
            "L_layers": 4,  # Low-level transformer layers
            
            # Model architecture
            "hidden_size": 512,
            "expansion": 2.0,
            "num_heads": 8,
            "pos_encodings": "rope",
            
            # Halting mechanism
            "halt_max_steps": 12,
            "halt_exploration_prob": 0.1,
            
            "forward_dtype": "float32"  # Use float32 for MPS compatibility
        }
        
        # Update with provided kwargs
        defaults.update(kwargs)
        
        super().__init__(**defaults)


class CodeSynthesisHRM(HierarchicalReasoningModel_ACTV1):
    """HRM model adapted for CodeSynthesis tasks."""
    
    def __init__(self, config: CodeSynthesisHRMConfig):
        # Convert config to dict for parent class
        config_dict = {
            "batch_size": config.batch_size,
            "seq_len": config.seq_len,
            "puzzle_emb_ndim": config.puzzle_emb_ndim,
            "num_puzzle_identifiers": config.num_puzzle_identifiers,
            "vocab_size": config.vocab_size,
            "H_cycles": config.H_cycles,
            "L_cycles": config.L_cycles,
            "H_layers": config.H_layers,
            "L_layers": config.L_layers,
            "hidden_size": config.hidden_size,
            "expansion": config.expansion,
            "num_heads": config.num_heads,
            "pos_encodings": config.pos_encodings,
            "rms_norm_eps": config.rms_norm_eps,
            "rope_theta": config.rope_theta,
            "halt_max_steps": config.halt_max_steps,
            "halt_exploration_prob": config.halt_exploration_prob,
            "forward_dtype": config.forward_dtype
        }
        
        super().__init__(config_dict)
        
        # Add task-specific output heads
        self.assumption_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.challenge_head = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.quality_head = torch.nn.Linear(config.hidden_size, 1)  # Quality score 0-1
        
        # Initialize weights
        trunc_normal_init_(self.assumption_head.weight)
        trunc_normal_init_(self.challenge_head.weight)
        trunc_normal_init_(self.quality_head.weight)
    
    def forward_task_specific(self, hidden_states: torch.Tensor, task_type: str) -> torch.Tensor:
        """Forward pass with task-specific output head."""
        
        if task_type == "assumption_detection":
            return self.assumption_head(hidden_states)
        elif task_type == "challenge_generation":
            return self.challenge_head(hidden_states)
        elif task_type == "quality_assessment":
            return self.quality_head(hidden_states)
        else:
            # Default to challenge generation
            return self.challenge_head(hidden_states)
    
    def analyze_code(self, code: str, task_type: str, tokenizer) -> Dict[str, Any]:
        """Analyze code with hierarchical reasoning."""
        
        # Tokenize input
        tokens = tokenizer.encode_code_analysis(code, task_type)
        
        with torch.no_grad():
            # Run hierarchical reasoning
            outputs = self.forward(tokens)
            
            # Get task-specific output
            task_output = self.forward_task_specific(outputs.last_hidden_state, task_type)
            
            # Decode results
            if task_type == "quality_assessment":
                quality_score = torch.sigmoid(task_output).item()
                return {"quality_score": quality_score}
            else:
                # For text generation tasks
                predicted_tokens = torch.argmax(task_output, dim=-1)
                decoded_text = tokenizer.decode(predicted_tokens)
                return {"generated_text": decoded_text}


class CodeSynthesisTokenizer:
    """Tokenizer for CodeSynthesis HRM model."""
    
    def __init__(self, vocab: Dict[str, int]):
        self.vocab = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        self.max_length = 512
    
    def encode_code_analysis(self, code: str, task_type: str) -> torch.Tensor:
        """Encode code analysis input."""
        
        tokens = ["<START>"]
        
        # Add task type
        tokens.extend(self._tokenize_text(task_type))
        tokens.append("<SEP>")
        
        # Add code
        tokens.extend(self._tokenize_code(code))
        
        # Convert to IDs
        token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        
        # Pad or truncate
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids.extend([self.vocab["<PAD>"]] * (self.max_length - len(token_ids)))
        
        return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        
        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze(0)
        
        tokens = []
        for token_id in token_ids:
            token = self.id_to_token.get(token_id.item(), "<UNK>")
            if token not in ["<PAD>", "<START>", "<END>"]:
                tokens.append(token)
        
        return " ".join(tokens)
    
    def _tokenize_code(self, code: str) -> List[str]:
        """Tokenize code."""
        import re
        tokens = re.findall(r'\w+|[^\w\s]', code)
        return [token.lower() for token in tokens if token.strip()]
    
    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text."""
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return [token for token in tokens if token.strip()]


def create_codesynthesis_hrm_model(dataset_path: str) -> Tuple[CodeSynthesisHRM, CodeSynthesisTokenizer]:
    """Create a CodeSynthesis HRM model and tokenizer."""
    
    # Load dataset to get vocabulary
    dataset = CodeSynthesisPuzzleDataset(dataset_path)
    
    # Create config
    config = CodeSynthesisHRMConfig(vocab_size=dataset.vocab_size)
    
    # Create model
    model = CodeSynthesisHRM(config)
    
    # Create tokenizer
    tokenizer = CodeSynthesisTokenizer(dataset.vocab)
    
    print(f"Created CodeSynthesis HRM model:")
    print(f"  - Vocabulary size: {dataset.vocab_size}")
    print(f"  - Hidden size: {config.hidden_size}")
    print(f"  - H/L cycles: {config.H_cycles}/{config.L_cycles}")
    print(f"  - H/L layers: {config.H_layers}/{config.L_layers}")
    
    return model, tokenizer


if __name__ == "__main__":
    # Test the adapter
    dataset_path = "data/codesynthesis-hrm/codesynthesis_train.json"
    
    if os.path.exists(dataset_path):
        model, tokenizer = create_codesynthesis_hrm_model(dataset_path)
        
        # Test code analysis
        test_code = '''
def process_data(data):
    result = data.get("value") * 2
    return result
'''
        
        print(f"\nTesting code analysis:")
        print(f"Code: {test_code}")
        
        analysis = model.analyze_code(test_code, "assumption_detection", tokenizer)
        print(f"Analysis: {analysis}")
        
    else:
        print(f"Dataset not found: {dataset_path}")
        print("Run build_codesynthesis_dataset.py first")