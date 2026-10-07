"""
Create Swift documentation-based training dataset for CodeSynthesis.
Uses official Swift language guide examples and patterns.
"""

import json
import os
from typing import List, Dict, Any


def create_swift_training_dataset():
    """Create training dataset based on Swift documentation examples."""
    
    print("🍎 Creating Swift Documentation-Based Training Dataset")
    print("=" * 60)
    
    # Swift code examples from official documentation with analysis
    swift_examples = [
        # Basic Syntax Examples
        {
            "code": '''
func greet(person: String) -> String {
    let greeting = "Hello, " + person + "!"
    return greeting
}
''',
            "assumptions": [
                "person parameter is not nil",
                "string concatenation will succeed",
                "function will always return a valid string"
            ],
            "challenges": [
                "What if person is an empty string?",
                "How would you localize the greeting?",
                "Should you validate the person parameter?",
                "How could you make this more efficient?"
            ],
            "quality_score": 0.6,
            "issues": ["No parameter validation", "String concatenation could be optimized"],
            "recommendations": ["Add guard statement for validation", "Use string interpolation"]
        },
        
        # Optionals and Safety
        {
            "code": '''
func findIndex(ofString valueToFind: String, in array: [String]) -> Int? {
    for (index, value) in array.enumerated() {
        if value == valueToFind {
            return index
        }
    }
    return nil
}
''',
            "assumptions": [
                "array parameter is not nil",
                "string comparison is case-sensitive",
                "array elements are valid strings"
            ],
            "challenges": [
                "Should this be case-insensitive?",
                "How would you handle Unicode normalization?",
                "What about performance with large arrays?",
                "Could you use built-in methods instead?"
            ],
            "quality_score": 0.8,
            "issues": [],
            "recommendations": ["Consider using firstIndex(of:) built-in method", "Add documentation"]
        },
        
        # Error Handling
        {
            "code": '''
enum VendingMachineError: Error {
    case invalidSelection
    case insufficientFunds(coinsNeeded: Int)
    case outOfStock
}

func vend(itemNamed name: String) throws {
    if name == "invalid" {
        throw VendingMachineError.invalidSelection
    }
}
''',
            "assumptions": [
                "name parameter is valid string",
                "caller will handle thrown errors",
                "error cases are properly defined"
            ],
            "challenges": [
                "How do you test error conditions?",
                "What other validation should be added?",
                "How would you log errors for debugging?",
                "Should you have a default error case?"
            ],
            "quality_score": 0.7,
            "issues": ["Incomplete implementation", "No input validation"],
            "recommendations": ["Complete the vending logic", "Add comprehensive error handling"]
        },
        
        # Classes and Inheritance
        {
            "code": '''
class Vehicle {
    var currentSpeed = 0.0
    var description: String {
        return "traveling at \\(currentSpeed) miles per hour"
    }
    func makeNoise() {
        // do nothing - an arbitrary vehicle doesn't necessarily make a noise
    }
}
''',
            "assumptions": [
                "speed is always non-negative",
                "speed unit is miles per hour",
                "description will be used for display"
            ],
            "challenges": [
                "Should speed be validated to prevent negative values?",
                "How would you handle different speed units?",
                "What about thread safety for currentSpeed?",
                "Should makeNoise be abstract/required?"
            ],
            "quality_score": 0.6,
            "issues": ["No speed validation", "Empty method implementation"],
            "recommendations": ["Add speed validation", "Consider protocol-based design"]
        },
        
        # Protocols and Extensions
        {
            "code": '''
protocol Drawable {
    func draw()
}

extension Array: Drawable where Element: Drawable {
    func draw() {
        for element in self {
            element.draw()
        }
    }
}
''',
            "assumptions": [
                "all elements conform to Drawable",
                "draw() method is safe to call on all elements",
                "order of drawing matters"
            ],
            "challenges": [
                "What if an element's draw() throws an error?",
                "How would you handle async drawing operations?",
                "Should there be a way to skip invalid elements?",
                "How about parallel drawing for performance?"
            ],
            "quality_score": 0.8,
            "issues": [],
            "recommendations": ["Add error handling for individual draws", "Consider async support"]
        },
        
        # Memory Management
        {
            "code": '''
class Person {
    let name: String
    var apartment: Apartment?
    
    init(name: String) {
        self.name = name
    }
}

class Apartment {
    let unit: String
    weak var tenant: Person?
    
    init(unit: String) {
        self.unit = unit
    }
}
''',
            "assumptions": [
                "weak reference prevents retain cycles",
                "name and unit are never empty",
                "relationships are properly managed"
            ],
            "challenges": [
                "How do you handle the tenant moving out?",
                "What about validation of unit numbers?",
                "Should name be changeable after creation?",
                "How would you handle multiple tenants?"
            ],
            "quality_score": 0.9,
            "issues": [],
            "recommendations": ["Add validation for name/unit", "Consider tenant change notifications"]
        },
        
        # Networking and Async
        {
            "code": '''
func fetchUserData(userID: String) async throws -> User {
    let url = URL(string: "https://api.example.com/users/\\(userID)")!
    let (data, _) = try await URLSession.shared.data(from: url)
    return try JSONDecoder().decode(User.self, from: data)
}
''',
            "assumptions": [
                "userID is valid and safe for URL",
                "network connection is available",
                "API returns valid JSON",
                "User struct matches API response"
            ],
            "challenges": [
                "How do you handle network timeouts?",
                "What about URL encoding for userID?",
                "Should you validate the HTTP response status?",
                "How would you add caching?",
                "What about rate limiting?"
            ],
            "quality_score": 0.4,
            "issues": ["Force unwrapping URL", "No error handling for network issues", "No input validation"],
            "recommendations": ["Proper URL validation", "Add network error handling", "Validate userID parameter"]
        },
        
        # SwiftUI Examples
        {
            "code": '''
struct ContentView: View {
    @State private var name = ""
    
    var body: some View {
        VStack {
            TextField("Enter your name", text: $name)
            Text("Hello, \\(name)!")
        }
        .padding()
    }
}
''',
            "assumptions": [
                "name binding works correctly",
                "UI updates happen on main thread",
                "TextField input is always valid"
            ],
            "challenges": [
                "How would you limit text input length?",
                "What about input validation for names?",
                "Should you handle empty names differently?",
                "How would you add accessibility support?"
            ],
            "quality_score": 0.7,
            "issues": ["No input validation", "Missing accessibility"],
            "recommendations": ["Add text field validation", "Include accessibility labels"]
        },
        
        # Core Data / Persistence
        {
            "code": '''
func saveContext() {
    let context = persistentContainer.viewContext
    
    if context.hasChanges {
        try context.save()
    }
}
''',
            "assumptions": [
                "persistentContainer is initialized",
                "context is available",
                "save operation will succeed"
            ],
            "challenges": [
                "What if the save operation fails?",
                "How do you handle merge conflicts?",
                "Should you save on a background queue?",
                "What about data validation before saving?"
            ],
            "quality_score": 0.3,
            "issues": ["No error handling", "Force try without catch", "No validation"],
            "recommendations": ["Add proper error handling with do-catch", "Validate data before saving"]
        },
        
        # Security Examples
        {
            "code": '''
func authenticateUser(username: String, password: String) -> Bool {
    let storedPassword = keychain.retrievePassword(for: username)
    return password == storedPassword
}
''',
            "assumptions": [
                "keychain access is secure",
                "password comparison is safe",
                "username exists in keychain"
            ],
            "challenges": [
                "How do you prevent timing attacks?",
                "What if keychain access fails?",
                "Should passwords be hashed?",
                "How do you handle failed authentication attempts?"
            ],
            "quality_score": 0.2,
            "issues": ["Vulnerable to timing attacks", "No error handling", "Plain text comparison"],
            "recommendations": ["Use secure password comparison", "Add proper error handling", "Implement rate limiting"]
        }
    ]
    
    # Convert to training format
    training_data = []
    
    for example in swift_examples:
        # Assumption detection task
        training_data.append({
            "input": {
                "task": "assumption_detection",
                "code": example["code"],
                "context": "Swift code analysis for assumption detection"
            },
            "output": [
                "Hidden assumptions detected:",
                *[f"• {assumption}" for assumption in example["assumptions"]]
            ]
        })
        
        # Challenge generation task
        training_data.append({
            "input": {
                "task": "challenge_generation",
                "code": example["code"],
                "context": "Swift code analysis for challenge questions"
            },
            "output": [
                "Challenge questions:",
                *[f"{i+1}. {challenge}" for i, challenge in enumerate(example["challenges"])]
            ]
        })
        
        # Quality assessment task
        quality_label = ("excellent" if example["quality_score"] > 0.8 else 
                        "good" if example["quality_score"] > 0.6 else
                        "medium" if example["quality_score"] > 0.4 else "low")
        
        quality_output = [f"Code quality assessment: {quality_label} ({example['quality_score']:.1f}/1.0)"]
        
        if example["issues"]:
            quality_output.append("")
            quality_output.append("Issues identified:")
            quality_output.extend([f"• {issue}" for issue in example["issues"]])
        
        if example["recommendations"]:
            quality_output.append("")
            quality_output.append("Recommendations:")
            quality_output.extend([f"• {rec}" for rec in example["recommendations"]])
        
        training_data.append({
            "input": {
                "task": "quality_assessment",
                "code": example["code"],
                "context": "Swift code quality assessment"
            },
            "output": quality_output
        })
    
    # Create training directory
    training_dir = "/Users/joel/GitHub/HRM/training"
    os.makedirs(training_dir, exist_ok=True)
    
    # Save Swift training dataset
    swift_train_path = os.path.join(training_dir, "swift_train.json")
    with open(swift_train_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    
    # Create dataset statistics
    stats = {
        "total_examples": len(training_data),
        "swift_code_snippets": len(swift_examples),
        "task_distribution": {
            "assumption_detection": len([x for x in training_data if x["input"]["task"] == "assumption_detection"]),
            "challenge_generation": len([x for x in training_data if x["input"]["task"] == "challenge_generation"]),
            "quality_assessment": len([x for x in training_data if x["input"]["task"] == "quality_assessment"])
        },
        "code_categories": {
            "basic_syntax": 1,
            "optionals_safety": 1,
            "error_handling": 1,
            "classes_inheritance": 1,
            "protocols_extensions": 1,
            "memory_management": 1,
            "networking_async": 1,
            "swiftui": 1,
            "persistence": 1,
            "security": 1
        },
        "quality_distribution": {
            "excellent": len([x for x in swift_examples if x["quality_score"] > 0.8]),
            "good": len([x for x in swift_examples if 0.6 < x["quality_score"] <= 0.8]),
            "medium": len([x for x in swift_examples if 0.4 < x["quality_score"] <= 0.6]),
            "low": len([x for x in swift_examples if x["quality_score"] <= 0.4])
        }
    }
    
    stats_path = os.path.join(training_dir, "swift_dataset_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Swift training dataset created!")
    print(f"   Training file: {swift_train_path}")
    print(f"   Statistics: {stats_path}")
    print(f"   Total examples: {len(training_data)}")
    print(f"   Swift code snippets: {len(swift_examples)}")
    print(f"   Task distribution:")
    for task, count in stats["task_distribution"].items():
        print(f"     {task}: {count}")
    print(f"   Code categories: {len(stats['code_categories'])} different Swift areas")
    print(f"   Quality distribution:")
    for quality, count in stats["quality_distribution"].items():
        print(f"     {quality}: {count} examples")
    
    return swift_train_path, stats


def create_additional_swift_patterns():
    """Create additional Swift patterns for comprehensive training."""
    
    additional_patterns = [
        # Concurrency patterns
        {
            "code": '''
actor BankAccount {
    private var balance: Double = 0.0
    
    func deposit(amount: Double) {
        guard amount > 0 else { return }
        balance += amount
    }
    
    func withdraw(amount: Double) -> Bool {
        guard amount > 0 && balance >= amount else { return false }
        balance -= amount
        return true
    }
}
''',
            "assumptions": [
                "actor isolation prevents data races",
                "balance operations are atomic",
                "amount validation is sufficient"
            ],
            "challenges": [
                "How do you handle overdraft scenarios?",
                "What about concurrent deposits and withdrawals?",
                "Should there be transaction logging?",
                "How would you add interest calculations?"
            ],
            "quality_score": 0.8,
            "issues": [],
            "recommendations": ["Add transaction history", "Consider decimal precision for currency"]
        },
        
        # Generic patterns
        {
            "code": '''
struct Stack<Element> {
    private var items: [Element] = []
    
    mutating func push(_ item: Element) {
        items.append(item)
    }
    
    mutating func pop() -> Element? {
        return items.popLast()
    }
}
''',
            "assumptions": [
                "array operations are efficient",
                "generic type is appropriate",
                "mutating methods are thread-safe in single-threaded context"
            ],
            "challenges": [
                "How would you make this thread-safe?",
                "What about stack overflow protection?",
                "Should you add a peek method?",
                "How would you implement a maximum capacity?"
            ],
            "quality_score": 0.7,
            "issues": ["No capacity limits", "Not thread-safe"],
            "recommendations": ["Add capacity management", "Consider thread safety if needed"]
        },
        
        # Result type patterns
        {
            "code": '''
func parseJSON<T: Codable>(_ data: Data, as type: T.Type) -> Result<T, Error> {
    do {
        let result = try JSONDecoder().decode(type, from: data)
        return .success(result)
    } catch {
        return .failure(error)
    }
}
''',
            "assumptions": [
                "data contains valid JSON",
                "type conforms to Codable properly",
                "decoder configuration is appropriate"
            ],
            "challenges": [
                "How would you customize the decoder configuration?",
                "What about handling different date formats?",
                "Should you validate the data before parsing?",
                "How would you add logging for failures?"
            ],
            "quality_score": 0.9,
            "issues": [],
            "recommendations": ["Consider custom decoder configuration", "Add specific error types"]
        }
    ]
    
    return additional_patterns


if __name__ == "__main__":
    swift_train_path, stats = create_swift_training_dataset()
    
    print("\\n🎯 Swift Training Dataset Summary:")
    print(f"   Ready for training with {stats['total_examples']} examples")
    print(f"   Covers {len(stats['code_categories'])} Swift language areas")
    print(f"   Balanced across assumption detection, challenges, and quality assessment")
    print(f"   Includes examples from basic syntax to advanced patterns")
    
    print("\\n📚 Next steps:")
    print("   1. Use this dataset to train the improved model")
    print("   2. Test on real Swift code")
    print("   3. Iterate based on performance")