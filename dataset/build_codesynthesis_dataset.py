"""
Code analysis dataset builder for HRM training.
Creates puzzle-format datasets from code review examples for training
HRM to understand code patterns and generate challenges.
"""

import json
import os
import ast
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import random


@dataclass
class CodeAnalysisPuzzle:
    """A puzzle for training HRM on code analysis tasks."""
    input_code: str
    task_type: str  # 'assumption_detection', 'challenge_generation', 'quality_assessment'
    expected_output: List[str]
    context: str
    difficulty: str  # 'basic', 'intermediate', 'advanced'
    metadata: Dict[str, Any]


class CodeSynthesisDatasetBuilder:
    """Builds HRM training datasets for CodeSynthesis tasks."""
    
    def __init__(self, output_dir: str = "data/codesynthesis-hrm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Patterns for different types of code issues
        self.assumption_patterns = [
            "assumes input is valid",
            "expects non-null values", 
            "relies on external service availability",
            "trusts user input without validation",
            "assumes network connectivity",
            "expects specific data format",
            "relies on third-party dependencies",
            "assumes single-threaded execution"
        ]
        
        self.challenge_templates = [
            "What happens if {condition}?",
            "How would you handle {scenario}?", 
            "What assumptions are made about {aspect}?",
            "What edge cases exist for {functionality}?",
            "How would this scale with {constraint}?",
            "What security concerns exist with {implementation}?"
        ]
    
    def generate_assumption_detection_puzzles(self, num_samples: int = 500) -> List[CodeAnalysisPuzzle]:
        """Generate puzzles for training assumption detection."""
        puzzles = []
        
        code_examples = self._get_code_examples_with_assumptions()
        
        for i in range(num_samples):
            code, assumptions = random.choice(code_examples)
            
            puzzle = CodeAnalysisPuzzle(
                input_code=code,
                task_type="assumption_detection",
                expected_output=assumptions,
                context="Identify hidden assumptions in this code",
                difficulty=self._assess_difficulty(code),
                metadata={
                    "language": "python",
                    "complexity_score": self._calculate_complexity(code),
                    "assumption_count": len(assumptions)
                }
            )
            puzzles.append(puzzle)
        
        return puzzles
    
    def generate_challenge_generation_puzzles(self, num_samples: int = 500) -> List[CodeAnalysisPuzzle]:
        """Generate puzzles for training challenge generation."""
        puzzles = []
        
        code_examples = self._get_code_examples_with_challenges()
        
        for i in range(num_samples):
            code, challenges = random.choice(code_examples)
            
            puzzle = CodeAnalysisPuzzle(
                input_code=code,
                task_type="challenge_generation", 
                expected_output=challenges,
                context="Generate challenging questions about this code",
                difficulty=self._assess_difficulty(code),
                metadata={
                    "language": "python",
                    "challenge_types": [self._categorize_challenge(c) for c in challenges],
                    "complexity_score": self._calculate_complexity(code)
                }
            )
            puzzles.append(puzzle)
        
        return puzzles
    
    def generate_quality_assessment_puzzles(self, num_samples: int = 300) -> List[CodeAnalysisPuzzle]:
        """Generate puzzles for training code quality assessment."""
        puzzles = []
        
        for i in range(num_samples):
            code, quality_score, issues = self._generate_code_with_quality_metrics()
            
            puzzle = CodeAnalysisPuzzle(
                input_code=code,
                task_type="quality_assessment",
                expected_output=[f"quality_score:{quality_score:.2f}"] + issues,
                context="Assess the quality of this code and identify issues",
                difficulty=self._assess_difficulty(code),
                metadata={
                    "language": "python",
                    "quality_score": quality_score,
                    "issue_count": len(issues)
                }
            )
            puzzles.append(puzzle)
        
        return puzzles
    
    def _get_code_examples_with_assumptions(self) -> List[Tuple[str, List[str]]]:
        """Generate code examples with their assumptions."""
        examples = [
            # Network/API assumptions
            ('''
def fetch_user_data(user_id):
    response = requests.get(f"https://api.example.com/users/{user_id}")
    return response.json()
''', [
                "assumes network connectivity is available",
                "trusts that API will return valid JSON",
                "expects user_id to be valid format",
                "assumes API endpoint is always accessible"
            ]),
            
            # Input validation assumptions
            ('''
def calculate_average(numbers):
    total = sum(numbers)
    return total / len(numbers)
''', [
                "assumes numbers list is not empty",
                "assumes all items in numbers are numeric",
                "assumes numbers parameter is iterable",
                "expects no None values in the list"
            ]),
            
            # Database assumptions
            ('''
def get_user_profile(user_id):
    user = db.users.find_one({"_id": user_id})
    return {
        "name": user["name"],
        "email": user["email"]
    }
''', [
                "assumes database connection is available",
                "trusts that user exists in database",
                "expects user document has name and email fields",
                "assumes user_id format matches database expectations"
            ]),
            
            # File handling assumptions
            ('''
def read_config():
    with open("config.json", "r") as f:
        return json.load(f)
''', [
                "assumes config.json file exists",
                "trusts file contains valid JSON",
                "expects read permissions on the file",
                "assumes current working directory contains the file"
            ]),
            
            # Type assumptions
            ('''
def format_price(amount):
    return f"${amount:.2f}"
''', [
                "assumes amount is numeric",
                "expects amount to be positive",
                "trusts amount has decimal precision",
                "assumes USD currency format is appropriate"
            ]),
            
            # Concurrency assumptions
            ('''
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count
''', [
                "assumes single-threaded access",
                "expects no concurrent modifications",
                "trusts that increment operations are atomic",
                "assumes no race conditions will occur"
            ])
        ]
        
        # Add variations and augmentations
        augmented_examples = []
        for code, assumptions in examples:
            augmented_examples.append((code.strip(), assumptions))
            
            # Create variations
            for i in range(3):  # 3 variations per example
                varied_code = self._create_code_variation(code)
                varied_assumptions = self._adapt_assumptions(assumptions, varied_code)
                augmented_examples.append((varied_code, varied_assumptions))
                
        return augmented_examples
    
    def _get_code_examples_with_challenges(self) -> List[Tuple[str, List[str]]]:
        """Generate code examples with challenging questions."""
        examples = [
            ('''
def process_payment(amount, card_number):
    if amount > 0:
        # Process payment
        result = payment_gateway.charge(card_number, amount)
        return result.success
    return False
''', [
                "What happens if amount is exactly zero or negative?",
                "How do you validate the card_number format?",
                "What if payment_gateway is unavailable?",
                "How do you handle partial payments or refunds?",
                "What security measures protect the card_number?",
                "How would you test this function safely?"
            ]),
            
            ('''
def cache_user_data(user_id, data):
    cache_key = f"user:{user_id}"
    cache.set(cache_key, data, timeout=3600)
    return cache_key
''', [
                "What if the cache service is down?",
                "How do you handle cache key collisions?",
                "What happens when the timeout expires?",
                "How large can the data be?",
                "What if user_id contains special characters?",
                "How do you invalidate stale cache entries?"
            ]),
            
            ('''
def parse_csv_file(filename):
    rows = []
    with open(filename, 'r') as f:
        for line in f:
            rows.append(line.strip().split(','))
    return rows
''', [
                "What if the file doesn't exist?",
                "How do you handle CSV files with quotes and commas?",
                "What about different encodings?",
                "How large files would break this approach?",
                "What if some rows have different column counts?",
                "How do you handle malformed CSV data?"
            ])
        ]
        
        # Augment with variations
        augmented_examples = []
        for code, challenges in examples:
            augmented_examples.append((code.strip(), challenges))
            
            # Create variations
            for i in range(2):
                varied_code = self._create_code_variation(code)
                adapted_challenges = self._adapt_challenges(challenges, varied_code)
                augmented_examples.append((varied_code, adapted_challenges))
        
        return augmented_examples
    
    def _generate_code_with_quality_metrics(self) -> Tuple[str, float, List[str]]:
        """Generate code with quality assessment."""
        code_templates = [
            # High quality code
            ('''
def validate_email(email: str) -> bool:
    """Validate email address format."""
    if not email or not isinstance(email, str):
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None
''', 0.9, ["Well-documented", "Type hints included", "Input validation"]),
            
            # Medium quality code
            ('''
def get_user_name(user_id):
    user = database.get_user(user_id)
    if user:
        return user.name
    return "Unknown"
''', 0.6, ["Missing error handling", "No type hints", "Assumes database connection"]),
            
            # Low quality code
            ('''
def calc(a, b, op):
    if op == '+':
        return a + b
    elif op == '-':
        return a - b
    elif op == '*':
        return a * b
    elif op == '/':
        return a / b
''', 0.3, ["No input validation", "Division by zero risk", "Poor function name", "No documentation"])
        ]
        
        return random.choice(code_templates)
    
    def _create_code_variation(self, code: str) -> str:
        """Create a variation of the given code."""
        # Simple variations - could be more sophisticated
        variations = [
            lambda c: c.replace("def ", "def new_"),
            lambda c: c.replace("user_id", "user_identifier"),
            lambda c: c.replace("amount", "value"),
            lambda c: re.sub(r'(\w+)\.(\w+)', r'\1.get_\2', c)  # method variations
        ]
        
        variation = random.choice(variations)
        try:
            return variation(code)
        except:
            return code  # Return original if variation fails
    
    def _adapt_assumptions(self, assumptions: List[str], code: str) -> List[str]:
        """Adapt assumptions to match code variations."""
        # Simple adaptation - could be more sophisticated
        adapted = []
        for assumption in assumptions:
            if "user_id" in assumption and "user_identifier" in code:
                adapted.append(assumption.replace("user_id", "user_identifier"))
            else:
                adapted.append(assumption)
        return adapted
    
    def _adapt_challenges(self, challenges: List[str], code: str) -> List[str]:
        """Adapt challenges to match code variations."""
        return challenges  # For now, keep challenges the same
    
    def _assess_difficulty(self, code: str) -> str:
        """Assess code difficulty based on complexity."""
        complexity = self._calculate_complexity(code)
        
        if complexity < 5:
            return "basic"
        elif complexity < 15:
            return "intermediate"
        else:
            return "advanced"
    
    def _calculate_complexity(self, code: str) -> int:
        """Calculate approximate code complexity."""
        try:
            tree = ast.parse(code)
            
            complexity = 0
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(node, ast.FunctionDef):
                    complexity += len(node.args.args)  # Add parameter count
                    
            return complexity + len(code.split('\n'))  # Add line count
        except:
            return len(code.split('\n'))  # Fallback to line count
    
    def _categorize_challenge(self, challenge: str) -> str:
        """Categorize a challenge question."""
        challenge_lower = challenge.lower()
        
        if any(word in challenge_lower for word in ['error', 'fail', 'exception']):
            return "error_handling"
        elif any(word in challenge_lower for word in ['security', 'validate', 'safe']):
            return "security"
        elif any(word in challenge_lower for word in ['scale', 'performance', 'large']):
            return "scalability"
        elif any(word in challenge_lower for word in ['test', 'mock', 'unit']):
            return "testing"
        else:
            return "general"
    
    def save_dataset(self, puzzles: List[CodeAnalysisPuzzle], dataset_name: str):
        """Save puzzles in HRM format."""
        
        # Convert to HRM puzzle format
        hrm_puzzles = []
        for puzzle in puzzles:
            hrm_puzzle = {
                "input": {
                    "code": puzzle.input_code,
                    "task": puzzle.task_type,
                    "context": puzzle.context
                },
                "output": puzzle.expected_output,
                "metadata": {
                    "difficulty": puzzle.difficulty,
                    **puzzle.metadata
                }
            }
            hrm_puzzles.append(hrm_puzzle)
        
        # Save to JSON file
        output_file = self.output_dir / f"{dataset_name}.json"
        with open(output_file, 'w') as f:
            json.dump(hrm_puzzles, f, indent=2)
        
        print(f"Saved {len(hrm_puzzles)} puzzles to {output_file}")
        
        # Also create a sample for inspection
        sample_file = self.output_dir / f"{dataset_name}_sample.json"
        with open(sample_file, 'w') as f:
            json.dump(hrm_puzzles[:5], f, indent=2)
        
        return output_file
    
    def build_complete_dataset(self, total_samples: int = 1000):
        """Build complete dataset for CodeSynthesis HRM training."""
        
        print("🔧 Building CodeSynthesis HRM Training Dataset...")
        
        # Generate different types of puzzles
        assumption_puzzles = self.generate_assumption_detection_puzzles(total_samples // 3)
        challenge_puzzles = self.generate_challenge_generation_puzzles(total_samples // 3)  
        quality_puzzles = self.generate_quality_assessment_puzzles(total_samples // 3)
        
        print(f"Generated {len(assumption_puzzles)} assumption detection puzzles")
        print(f"Generated {len(challenge_puzzles)} challenge generation puzzles")
        print(f"Generated {len(quality_puzzles)} quality assessment puzzles")
        
        # Save individual datasets
        self.save_dataset(assumption_puzzles, "assumption_detection")
        self.save_dataset(challenge_puzzles, "challenge_generation")
        self.save_dataset(quality_puzzles, "quality_assessment")
        
        # Combine all puzzles
        all_puzzles = assumption_puzzles + challenge_puzzles + quality_puzzles
        random.shuffle(all_puzzles)
        
        # Split into train/test
        split_point = int(0.8 * len(all_puzzles))
        train_puzzles = all_puzzles[:split_point]
        test_puzzles = all_puzzles[split_point:]
        
        train_file = self.save_dataset(train_puzzles, "codesynthesis_train")
        test_file = self.save_dataset(test_puzzles, "codesynthesis_test")
        
        print(f"\n✅ Complete dataset created:")
        print(f"   Training: {len(train_puzzles)} puzzles -> {train_file}")
        print(f"   Testing: {len(test_puzzles)} puzzles -> {test_file}")
        
        # Generate dataset statistics
        stats = self._generate_dataset_stats(all_puzzles)
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"   Statistics: {stats_file}")
        
        return train_file, test_file
    
    def _generate_dataset_stats(self, puzzles: List[CodeAnalysisPuzzle]) -> Dict[str, Any]:
        """Generate statistics about the dataset."""
        
        task_counts = {}
        difficulty_counts = {}
        complexity_scores = []
        
        for puzzle in puzzles:
            # Task type distribution
            task_counts[puzzle.task_type] = task_counts.get(puzzle.task_type, 0) + 1
            
            # Difficulty distribution  
            difficulty_counts[puzzle.difficulty] = difficulty_counts.get(puzzle.difficulty, 0) + 1
            
            # Complexity scores
            if 'complexity_score' in puzzle.metadata:
                complexity_scores.append(puzzle.metadata['complexity_score'])
        
        stats = {
            "total_puzzles": len(puzzles),
            "task_distribution": task_counts,
            "difficulty_distribution": difficulty_counts,
            "average_complexity": sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            "complexity_range": [min(complexity_scores), max(complexity_scores)] if complexity_scores else [0, 0]
        }
        
        return stats


if __name__ == "__main__":
    builder = CodeSynthesisDatasetBuilder()
    train_file, test_file = builder.build_complete_dataset(total_samples=1000)
    
    print(f"\n🚀 Dataset ready for HRM training!")
    print(f"Next steps:")
    print(f"1. Review sample files to verify quality")
    print(f"2. Adjust HRM config to use: {train_file}")
    print(f"3. Run HRM training with CodeSynthesis-specific data")