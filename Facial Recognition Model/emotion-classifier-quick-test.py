import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import json
import time
from datetime import datetime

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load and preprocess data
print("Loading data...")
data = pd.read_csv('data/unity_collection.csv')

X = data.drop('Emotion', axis=1).values
y = data['Emotion'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
num_classes = len(set(y_encoded))

# Quick test configurations (most promising ones)
quick_configs = [
    {
        'name': 'Simple_ReLU',
        'layers': [128, 64],
        'activation': nn.ReLU(),
        'dropout': 0.2,
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 25
    },
    {
        'name': 'Deep_ReLU',
        'layers': [256, 128, 64],
        'activation': nn.ReLU(),
        'dropout': 0.3,
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 30
    },
    {
        'name': 'Wide_ReLU',
        'layers': [512, 256],
        'activation': nn.ReLU(),
        'dropout': 0.3,
        'optimizer': 'Adam',
        'lr': 0.0005,
        'epochs': 25
    },
    {
        'name': 'Simple_LeakyReLU',
        'layers': [128, 64],
        'activation': nn.LeakyReLU(0.1),
        'dropout': 0.2,
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 25
    },
    {
        'name': 'Deep_LeakyReLU',
        'layers': [256, 128, 64],
        'activation': nn.LeakyReLU(0.1),
        'dropout': 0.3,
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 30
    },
    {
        'name': 'Simple_ELU',
        'layers': [128, 64],
        'activation': nn.ELU(),
        'dropout': 0.2,
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 25
    },
    {
        'name': 'BatchNorm_Deep',
        'layers': [256, 128, 64],
        'activation': nn.ReLU(),
        'dropout': 0.2,
        'batch_norm': True,
        'optimizer': 'Adam',
        'lr': 0.001,
        'epochs': 25
    },
    {
        'name': 'SGD_Deep',
        'layers': [256, 128, 64],
        'activation': nn.ReLU(),
        'dropout': 0.2,
        'optimizer': 'SGD',
        'lr': 0.01,
        'epochs': 40
    }
]

class FlexibleModel(nn.Module):
    def __init__(self, input_size, num_classes, layers, activation, dropout, batch_norm=False):
        super(FlexibleModel, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if batch_norm else None
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        
        # Build layers
        prev_size = input_size
        for layer_size in layers:
            self.layers.append(nn.Linear(prev_size, layer_size))
            if batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_size))
            prev_size = layer_size
        
        # Output layer
        self.output = nn.Linear(prev_size, num_classes)
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.batch_norms is not None:
                x = self.batch_norms[i](x)
            x = self.activation(x)
            x = self.dropout(x)
        
        x = self.output(x)
        return x

def quick_test():
    results = []
    best_accuracy = 0
    best_model = None
    best_config = None
    
    print(f"Testing {len(quick_configs)} promising configurations...")
    
    for i, config in enumerate(quick_configs, 1):
        print(f"\nTest {i}/{len(quick_configs)}: {config['name']}")
        print(f"  Layers: {config['layers']}")
        print(f"  Optimizer: {config['optimizer']}, LR: {config['lr']}")
        
        start_time = time.time()
        
        # Create model
        model = FlexibleModel(
            input_size=input_size,
            num_classes=num_classes,
            layers=config['layers'],
            activation=config['activation'],
            dropout=config['dropout'],
            batch_norm=config.get('batch_norm', False)
        ).to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=config['lr'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
        
        # Create data loaders
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Training
        model.train()
        for epoch in range(config['epochs']):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch {epoch+1}/{config['epochs']}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        training_time = time.time() - start_time
        
        result = {
            'name': config['name'],
            'layers': config['layers'],
            'activation': str(config['activation']),
            'dropout': config['dropout'],
            'batch_norm': config.get('batch_norm', False),
            'optimizer': config['optimizer'],
            'learning_rate': config['lr'],
            'epochs': config['epochs'],
            'accuracy': accuracy,
            'f1_score': f1,
            'training_time': training_time
        }
        
        results.append(result)
        
        print(f"  Results: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
        
        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_config = result.copy()
            
            # Save best model
            model_filename = f"best_model_{accuracy:.4f}_{config['name']}.pth"
            torch.save(model.state_dict(), model_filename)
            print(f"  NEW BEST MODEL SAVED: {model_filename}")
            
            # Export to ONNX
            dummy_input = torch.randn(1, input_size).to(device)
            onnx_filename = f"best_model_{accuracy:.4f}_{config['name']}.onnx"
            torch.onnx.export(model, dummy_input, onnx_filename, 
                            input_names=['input'], output_names=['output'],
                            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
    
    return results, best_model, best_config

def save_quick_results(results, best_config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    results_filename = f"quick_test_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    df_results = pd.DataFrame(results)
    summary_filename = f"quick_test_summary_{timestamp}.csv"
    df_results.to_csv(summary_filename, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_filename}")
    print(f"  Summary: {summary_filename}")
    
    # Print ranking
    print(f"\nModel Ranking by Accuracy:")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:2d}. {result['name']} - Acc: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
    
    print(f"\nBest Model: {best_config['name']}")
    print(f"Best Accuracy: {best_config['accuracy']:.4f}")
    print(f"Best F1 Score: {best_config['f1_score']:.4f}")

if __name__ == "__main__":
    print("Starting Quick Emotion Classifier Testing")
    print("=" * 50)
    
    results, best_model, best_config = quick_test()
    save_quick_results(results, best_config)
    
    print("\nQuick testing completed!")
    print(f"Best model: {best_config['name']} with accuracy: {best_config['accuracy']:.4f}")
