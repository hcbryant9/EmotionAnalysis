import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.onnx
from torch.utils.data import DataLoader, TensorDataset
import itertools
import json
import time
from datetime import datetime
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Load and preprocess data
print("Loading data...")
data = pd.read_csv('data/unity_collection.csv')

X = data.drop('Emotion', axis=1).values
y = data['Emotion'].values

# Encode string labels to integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Data scaling options
scalers = {
    'none': None,
    'standard': StandardScaler()
}

# Define different model architectures
class SimpleModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=64, dropout=0.2):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

class DeepModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64, 32], dropout=0.2):
        super(DeepModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class WideModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_size=512, dropout=0.3):
        super(WideModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

class BatchNormModel(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64], dropout=0.2):
        super(BatchNormModel, self).__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# Define model configurations
model_configs = [
    {
        'name': 'Simple_64',
        'class': SimpleModel,
        'params': {'hidden_size': 64, 'dropout': 0.2}
    },
    {
        'name': 'Simple_128',
        'class': SimpleModel,
        'params': {'hidden_size': 128, 'dropout': 0.2}
    },
    {
        'name': 'Deep_3Layer',
        'class': DeepModel,
        'params': {'hidden_sizes': [128, 64, 32], 'dropout': 0.2}
    },
    {
        'name': 'Deep_4Layer',
        'class': DeepModel,
        'params': {'hidden_sizes': [256, 128, 64, 32], 'dropout': 0.3}
    },
    {
        'name': 'Wide_512',
        'class': WideModel,
        'params': {'hidden_size': 512, 'dropout': 0.3}
    },
    {
        'name': 'Wide_256',
        'class': WideModel,
        'params': {'hidden_size': 256, 'dropout': 0.2}
    },
    {
        'name': 'BatchNorm_2Layer',
        'class': BatchNormModel,
        'params': {'hidden_sizes': [128, 64], 'dropout': 0.2}
    },
    {
        'name': 'BatchNorm_3Layer',
        'class': BatchNormModel,
        'params': {'hidden_sizes': [256, 128, 64], 'dropout': 0.3}
    }
]

# Training configurations
training_configs = [
    {
        'optimizer': 'Adam',
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 20,
        'weight_decay': 0.0
    },
    {
        'optimizer': 'Adam',
        'lr': 0.0005,
        'batch_size': 32,
        'epochs': 30,
        'weight_decay': 0.001
    },
    {
        'optimizer': 'Adam',
        'lr': 0.002,
        'batch_size': 64,
        'epochs': 25,
        'weight_decay': 0.0001
    },
    {
        'optimizer': 'SGD',
        'lr': 0.01,
        'batch_size': 32,
        'epochs': 40,
        'weight_decay': 0.0001
    },
    {
        'optimizer': 'SGD',
        'lr': 0.001,
        'batch_size': 64,
        'epochs': 50,
        'weight_decay': 0.001
    },
    {
        'optimizer': 'RMSprop',
        'lr': 0.001,
        'batch_size': 32,
        'epochs': 30,
        'weight_decay': 0.0001
    }
]

def create_optimizer(model, config):
    if config['optimizer'] == 'Adam':
        return optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        return optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9, weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

def train_model(model, train_loader, test_loader, config):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, config)
    
    # Training loop
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
    
    return accuracy, f1, model

def run_mass_testing():
    input_size = X_train.shape[1]
    num_classes = len(set(y_encoded))
    
    results = []
    best_accuracy = 0
    best_model = None
    best_config = None
    
    print(f"Starting mass testing with {len(model_configs)} models and {len(training_configs)} training configs...")
    print(f"Total combinations: {len(model_configs) * len(training_configs) * len(scalers)}")
    
    experiment_count = 0
    total_experiments = len(model_configs) * len(training_configs) * len(scalers)
    
    for scaler_name, scaler in scalers.items():
        # Prepare data with scaling
        if scaler is not None:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        for model_config in model_configs:
            for training_config in training_configs:
                experiment_count += 1
                print(f"\nExperiment {experiment_count}/{total_experiments}")
                print(f"Model: {model_config['name']}, Optimizer: {training_config['optimizer']}, "
                      f"LR: {training_config['lr']}, Scaler: {scaler_name}")
                
                start_time = time.time()
                
                # Create data loaders
                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=training_config['batch_size'], shuffle=False)
                
                # Create model
                model = model_config['class'](input_size, num_classes, **model_config['params'])
                
                try:
                    # Train and evaluate
                    accuracy, f1, trained_model = train_model(model, train_loader, test_loader, training_config)
                    
                    training_time = time.time() - start_time
                    
                    # Store results
                    result = {
                        'model_name': model_config['name'],
                        'scaler': scaler_name,
                        'optimizer': training_config['optimizer'],
                        'learning_rate': training_config['lr'],
                        'batch_size': training_config['batch_size'],
                        'epochs': training_config['epochs'],
                        'weight_decay': training_config['weight_decay'],
                        'accuracy': accuracy,
                        'f1_score': f1,
                        'training_time': training_time,
                        'model_params': model_config['params']
                    }
                    
                    results.append(result)
                    
                    print(f"    Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
                    
                    # Track best model
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = trained_model
                        best_config = result.copy()
                        
                        # Save best model
                        model_filename = f"best_model_{accuracy:.4f}_{model_config['name']}.pth"
                        torch.save(trained_model.state_dict(), model_filename)
                        print(f"    NEW BEST MODEL SAVED: {model_filename}")
                        
                        # Export to ONNX
                        dummy_input = torch.randn(1, input_size).to(device)
                        onnx_filename = f"best_model_{accuracy:.4f}_{model_config['name']}.onnx"
                        torch.onnx.export(trained_model, dummy_input, onnx_filename, 
                                        input_names=['input'], output_names=['output'],
                                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                        
                except Exception as e:
                    print(f"    ERROR: {str(e)}")
                    continue
    
    return results, best_model, best_config

def save_results(results, best_config):
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"mass_testing_results_{timestamp}.json"
    
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    df_results = pd.DataFrame(results)
    summary_filename = f"mass_testing_summary_{timestamp}.csv"
    df_results.to_csv(summary_filename, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_filename}")
    print(f"  Summary: {summary_filename}")
    
    # Print top 10 models
    print(f"\nTop 10 Models by Accuracy:")
    top_models = df_results.nlargest(10, 'accuracy')
    for i, (_, row) in enumerate(top_models.iterrows(), 1):
        print(f"{i:2d}. {row['model_name']} ({row['scaler']}) - "
              f"Acc: {row['accuracy']:.4f}, F1: {row['f1_score']:.4f}")
    
    print(f"\nBest Model Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    print("Starting Emotion Classifier Mass Testing")
    print("=" * 50)
    
    results, best_model, best_config = run_mass_testing()
    save_results(results, best_config)
    
    print("\nMass testing completed!")
    print(f"Best accuracy achieved: {best_config['accuracy']:.4f}")
