"""
Enhanced CodeSynthesis model with improved vocabulary, architecture, and training data.
Addresses the key weaknesses identified in testing.
"""

import torch
import torch.nn as nn
import json
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np


@dataclass
class ImprovedModelConfig:
    """Configuration for improved CodeSynthesis model."""
    vocab_size: int = 500  # Expanded from 30
    embedding_dim: int = 256  # Increased from 128
    num_heads: int = 8
    num_layers: int = 6  # Increased from 4
    dropout: float = 0.1
    max_length: int = 256  # Increased from 128


class EnhancedCodeAnalysisModel(nn.Module):
    """Improved model with larger vocabulary and better architecture."""
    
    def __init__(self, config: ImprovedModelConfig):
        super().__init__()
        self.config = config
        
        # Enhanced embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_encoding = nn.Parameter(
            torch.randn(config.max_length, config.embedding_dim)
        )
        
        # Improved transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config.num_layers)
        
        # Specialized heads with more capacity
        self.assumption_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.embedding_dim)
        )
        
        self.challenge_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.embedding_dim)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single quality score
        )
        
        # Vulnerability detection head (NEW)
        self.vulnerability_head = nn.Sequential(
            nn.Linear(config.embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # 10 common vulnerability types
        )
        
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size)
        
    def forward(self, input_ids, task_type="assumption_detection"):
        # Enhanced embedding with positional encoding
        seq_len = input_ids.size(1)
        embeddings = self.embedding(input_ids)
        embeddings += self.positional_encoding[:seq_len].unsqueeze(0)
        
        # Transform with attention mask for padding
        padding_mask = (input_ids == 0)  # Assuming 0 is PAD token
        hidden_states = self.transformer(embeddings, src_key_padding_mask=padding_mask)
        
        # Use CLS token representation (first token) for classification tasks
        cls_hidden = hidden_states[:, 0, :]
        
        # Task-specific output
        if task_type == "assumption_detection":
            return self.assumption_head(cls_hidden)
        elif task_type == "challenge_generation":
            return self.challenge_head(cls_hidden)
        elif task_type == "quality_assessment":
            return torch.sigmoid(self.quality_head(cls_hidden))
        elif task_type == "vulnerability_detection":
            return torch.sigmoid(self.vulnerability_head(cls_hidden))
        else:
            return self.output_projection(hidden_states)


def build_enhanced_vocabulary(data_path: str) -> Dict[str, int]:
    """Build comprehensive vocabulary from training data and code patterns."""
    
    # Load training data to extract vocabulary
    with open(data_path, 'r') as f:
        puzzles = json.load(f)
    
    # Start with special tokens
    vocab = {
        "<PAD>": 0, "<UNK>": 1, "<CLS>": 2, "<SEP>": 3,
        "<START>": 4, "<END>": 5
    }
    
    # Python keywords and operators
    python_keywords = [
        "def", "class", "import", "from", "as", "if", "elif", "else", 
        "for", "while", "try", "except", "finally", "with", "return",
        "yield", "break", "continue", "pass", "global", "nonlocal",
        "lambda", "and", "or", "not", "in", "is", "None", "True", "False"
    ]
    
    # Common code patterns
    code_patterns = [
        # Data types
        "str", "int", "float", "bool", "list", "dict", "tuple", "set",
        # Common functions
        "len", "range", "enumerate", "zip", "map", "filter", "sum", "max", "min",
        # File operations
        "open", "read", "write", "close", "file", "path", "exists",
        # Database patterns
        "database", "db", "query", "select", "insert", "update", "delete",
        "connection", "cursor", "fetch", "commit", "rollback",
        # Web patterns
        "request", "response", "get", "post", "put", "delete", "api",
        "json", "xml", "html", "http", "https", "url", "endpoint",
        # Security patterns
        "password", "hash", "encrypt", "decrypt", "token", "auth", "login",
        "session", "cookie", "csrf", "xss", "sql", "injection",
        # Error patterns
        "error", "exception", "raise", "catch", "handle", "log", "debug",
        # Quality indicators
        "docstring", "comment", "type", "hint", "annotation", "test",
        # Operators and punctuation
        "(", ")", "[", "]", "{", "}", "=", "+", "-", "*", "/", "//", "%", "**",
        "==", "!=", "<", ">", "<=", ">=", ":", ";", ",", ".", "_", "'"
    ]
    
    # Vulnerability-specific terms
    vulnerability_terms = [
        "validate", "sanitize", "escape", "bounds", "overflow", "underflow",
        "null", "empty", "malicious", "attack", "exploit", "vulnerability",
        "secure", "unsafe", "risk", "threat", "buffer", "memory", "leak"
    ]
    
    # Code quality terms
    quality_terms = [
        "readable", "maintainable", "efficient", "clean", "documented",
        "tested", "modular", "reusable", "scalable", "performance",
        "complexity", "refactor", "optimize", "best", "practice"
    ]
    
    # Assumption-related terms
    assumption_terms = [
        "assume", "expect", "require", "depend", "valid", "exists", "available",
        "initialized", "connected", "loaded", "configured", "accessible"
    ]
    
    # Challenge-related terms
    challenge_terms = [
        "what", "if", "how", "why", "when", "where", "could", "should", "would",
        "might", "consider", "handle", "edge", "case", "scenario", "fail"
    ]
    
    # Combine all vocabulary
    all_terms = (python_keywords + code_patterns + vulnerability_terms + 
                quality_terms + assumption_terms + challenge_terms)
    
    # Add to vocabulary
    for term in all_terms:
        if term not in vocab:
            vocab[term] = len(vocab)
    
    # Extract terms from training data
    text_data = []
    for puzzle in puzzles:
        text_data.append(puzzle["input"]["code"])
        # Handle different output formats
        if isinstance(puzzle["output"], list):
            text_data.extend(puzzle["output"])
        elif isinstance(puzzle["output"], dict):
            text_data.append(puzzle["output"].get("response", ""))
        else:
            text_data.append(str(puzzle["output"]))
    
    # Tokenize all text and find frequent terms
    all_tokens = []
    for text in text_data:
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        all_tokens.extend(tokens)
    
    # Count frequencies and add frequent tokens
    from collections import Counter
    token_counts = Counter(all_tokens)
    
    for token, count in token_counts.most_common(200):  # Top 200 frequent tokens
        if token not in vocab and count >= 3:  # Only tokens appearing 3+ times
            vocab[token] = len(vocab)
            
            if len(vocab) >= 500:  # Cap vocabulary size
                break
    
    print(f"Built enhanced vocabulary with {len(vocab)} tokens")
    return vocab


def enhanced_tokenize(text: str, vocab: Dict[str, int], max_length: int = 256) -> torch.Tensor:
    """Enhanced tokenization with better code pattern recognition."""
    
    # Add CLS token at start
    tokens = ["<CLS>"]
    
    # Improved tokenization that preserves code structure
    # First, split on whitespace to preserve words
    words = text.split()
    
    for word in words:
        # Split each word on punctuation but keep punctuation
        parts = re.findall(r'\w+|[^\w\s]', word.lower())
        tokens.extend(parts)
    
    # Convert to IDs
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_length-1]]
    
    # Add SEP token
    token_ids.append(vocab["<SEP>"])
    
    # Pad to max length
    while len(token_ids) < max_length:
        token_ids.append(vocab["<PAD>"])
    
    return torch.tensor(token_ids[:max_length], dtype=torch.long)


def create_enhanced_training_data(base_data_path: str, vocab: Dict[str, int]) -> List[Dict]:
    """Create enhanced training data with vulnerability detection."""
    
    with open(base_data_path, 'r') as f:
        base_puzzles = json.load(f)
    
    enhanced_puzzles = []
    
    # Add vulnerability detection examples
    vulnerability_examples = [
        {
            "code": "def get_user_data(user_id): return users[user_id]",
            "vulnerabilities": ["array_bounds", "no_validation"],
            "assumptions": ["user_id exists in users", "users is not empty"],
            "quality_score": 0.2
        },
        {
            "code": "def authenticate(user, pwd): return user.password == pwd",
            "vulnerabilities": ["timing_attack", "plain_text_comparison"],
            "assumptions": ["user object exists", "password field exists"],
            "quality_score": 0.1
        },
        {
            "code": "def upload_file(data, name): open(f'/uploads/{name}', 'wb').write(data)",
            "vulnerabilities": ["path_traversal", "no_validation", "resource_leak"],
            "assumptions": ["upload directory exists", "write permissions", "data is valid"],
            "quality_score": 0.15
        },
        # Add more high-quality examples
        {
            "code": '''
def validate_email(email: str) -> bool:
    """Validate email format with comprehensive checks."""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
''',
            "vulnerabilities": [],
            "assumptions": ["re module imported"],
            "quality_score": 0.9
        }
    ]
    
    # Convert vulnerability examples to training format
    for example in vulnerability_examples:
        # Assumption detection puzzle
        enhanced_puzzles.append({
            "input": {
                "task": "assumption_detection",
                "code": example["code"]
            },
            "output": {
                "response": f"Assumptions detected: {', '.join(example['assumptions'])}"
            }
        })
        
        # Vulnerability detection puzzle (NEW)
        if example["vulnerabilities"]:
            enhanced_puzzles.append({
                "input": {
                    "task": "vulnerability_detection", 
                    "code": example["code"]
                },
                "output": {
                    "response": f"Vulnerabilities found: {', '.join(example['vulnerabilities'])}"
                }
            })
        
        # Quality assessment puzzle with better calibration
        quality_label = "high" if example["quality_score"] > 0.7 else "medium" if example["quality_score"] > 0.4 else "low"
        enhanced_puzzles.append({
            "input": {
                "task": "quality_assessment",
                "code": example["code"]
            },
            "output": {
                "response": f"Code quality: {quality_label} ({example['quality_score']:.1f}/1.0)"
            }
        })
    
    # Add original puzzles
    enhanced_puzzles.extend(base_puzzles)
    
    print(f"Enhanced training data: {len(enhanced_puzzles)} total puzzles")
    return enhanced_puzzles


if __name__ == "__main__":
    print("🔧 Building Enhanced CodeSynthesis Model")
    print("=" * 50)
    
    # Build enhanced vocabulary
    data_path = "/Users/joel/GitHub/HRM/data/codesynthesis-hrm/codesynthesis_train.json"
    vocab = build_enhanced_vocabulary(data_path)
    
    # Save vocabulary
    vocab_path = "/Users/joel/GitHub/HRM/data/codesynthesis-hrm/enhanced_vocab.json"
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Create enhanced training data
    enhanced_data = create_enhanced_training_data(data_path, vocab)
    
    enhanced_data_path = "/Users/joel/GitHub/HRM/data/codesynthesis-hrm/enhanced_train.json"
    with open(enhanced_data_path, 'w') as f:
        json.dump(enhanced_data, f, indent=2)
    
    # Create improved model config
    config = ImprovedModelConfig(vocab_size=len(vocab))
    
    print(f"✅ Enhanced model ready:")
    print(f"   Vocabulary size: {len(vocab)} (was 30)")
    print(f"   Embedding dim: {config.embedding_dim} (was 128)")
    print(f"   Layers: {config.num_layers} (was 4)")
    print(f"   Max length: {config.max_length} (was 128)")
    print(f"   Training examples: {len(enhanced_data)} (was 999)")
    print(f"   New capabilities: Vulnerability detection, Better quality scoring")