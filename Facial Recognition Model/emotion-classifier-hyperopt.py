import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import json
import time
from datetime import datetime
import random
from itertools import product

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

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

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

input_size = X_train.shape[1]
num_classes = len(set(y_encoded))

# Hyperparameter search space
hyperparameter_space = {
    'architecture': [
        {'layers': [64], 'name': 'shallow_64'},
        {'layers': [128], 'name': 'shallow_128'},
        {'layers': [256], 'name': 'shallow_256'},
        {'layers': [128, 64], 'name': 'medium_128_64'},
        {'layers': [256, 128], 'name': 'medium_256_128'},
        {'layers': [512, 256], 'name': 'wide_512_256'},
        {'layers': [256, 128, 64], 'name': 'deep_256_128_64'},
        {'layers': [512, 256, 128], 'name': 'deep_512_256_128'},
        {'layers': [256, 128, 64, 32], 'name': 'very_deep_256_128_64_32'},
    ],
    'activation': [
        {'func': nn.ReLU(), 'name': 'relu'},
        {'func': nn.LeakyReLU(0.1), 'name': 'leaky_relu'},
        {'func': nn.ELU(), 'name': 'elu'},
        {'func': nn.GELU(), 'name': 'gelu'},
    ],
    'dropout': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    'batch_norm': [True, False],
    'optimizer': [
        {'name': 'Adam', 'lr': [0.0001, 0.0005, 0.001, 0.002, 0.005]},
        {'name': 'SGD', 'lr': [0.001, 0.01, 0.1], 'momentum': [0.9, 0.95]},
        {'name': 'RMSprop', 'lr': [0.0001, 0.001, 0.01]},
    ],
    'batch_size': [16, 32, 64, 128],
    'weight_decay': [0.0, 0.0001, 0.001, 0.01],
    'scheduler': [
        {'name': 'none'},
        {'name': 'StepLR', 'step_size': 10, 'gamma': 0.1},
        {'name': 'ExponentialLR', 'gamma': 0.95},
        {'name': 'CosineAnnealingLR', 'T_max': 50},
    ]
}

class AdvancedModel(nn.Module):
    def __init__(self, input_size, num_classes, layers, activation, dropout, batch_norm=False):
        super(AdvancedModel, self).__init__()
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

def create_optimizer(model, opt_config, weight_decay):
    if opt_config['name'] == 'Adam':
        return optim.Adam(model.parameters(), lr=opt_config['lr'], weight_decay=weight_decay)
    elif opt_config['name'] == 'SGD':
        return optim.SGD(model.parameters(), lr=opt_config['lr'], 
                        momentum=opt_config.get('momentum', 0.9), weight_decay=weight_decay)
    elif opt_config['name'] == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=opt_config['lr'], weight_decay=weight_decay)

def create_scheduler(optimizer, scheduler_config):
    if scheduler_config['name'] == 'none':
        return None
    elif scheduler_config['name'] == 'StepLR':
        return lr_scheduler.StepLR(optimizer, step_size=scheduler_config['step_size'], 
                                 gamma=scheduler_config['gamma'])
    elif scheduler_config['name'] == 'ExponentialLR':
        return lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_config['gamma'])
    elif scheduler_config['name'] == 'CosineAnnealingLR':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config['T_max'])

def random_search(num_trials=100, epochs=30):
    """Random search for hyperparameter optimization"""
    results = []
    best_accuracy = 0
    best_model = None
    best_config = None
    
    print(f"Starting random search with {num_trials} trials...")
    
    for trial in range(num_trials):
        print(f"\nTrial {trial + 1}/{num_trials}")
        
        # Sample hyperparameters
        arch = random.choice(hyperparameter_space['architecture'])
        activation = random.choice(hyperparameter_space['activation'])
        dropout = random.choice(hyperparameter_space['dropout'])
        batch_norm = random.choice(hyperparameter_space['batch_norm'])
        batch_size = random.choice(hyperparameter_space['batch_size'])
        weight_decay = random.choice(hyperparameter_space['weight_decay'])
        scheduler_config = random.choice(hyperparameter_space['scheduler'])
        
        # Sample optimizer
        opt_type = random.choice(hyperparameter_space['optimizer'])
        lr = random.choice(opt_type['lr'])
        opt_config = {'name': opt_type['name'], 'lr': lr}
        if opt_type['name'] == 'SGD' and 'momentum' in opt_type:
            opt_config['momentum'] = random.choice(opt_type['momentum'])
        
        config = {
            'architecture': arch['name'],
            'layers': arch['layers'],
            'activation': activation['name'],
            'dropout': dropout,
            'batch_norm': batch_norm,
            'optimizer': opt_config['name'],
            'learning_rate': lr,
            'batch_size': batch_size,
            'weight_decay': weight_decay,
            'scheduler': scheduler_config['name'],
            'epochs': epochs
        }
        
        print(f"  Config: {arch['name']}, {activation['name']}, dropout={dropout}, "
              f"batch_norm={batch_norm}, {opt_config['name']}, lr={lr}")
        
        try:
            start_time = time.time()
            
            # Create model
            model = AdvancedModel(
                input_size=input_size,
                num_classes=num_classes,
                layers=arch['layers'],
                activation=activation['func'],
                dropout=dropout,
                batch_norm=batch_norm
            ).to(device)
            
            # Create optimizer and scheduler
            optimizer = create_optimizer(model, opt_config, weight_decay)
            scheduler = create_scheduler(optimizer, scheduler_config)
            
            # Create data loaders
            train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            # Training
            criterion = nn.CrossEntropyLoss()
            model.train()
            
            for epoch in range(epochs):
                total_loss = 0
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if scheduler:
                    scheduler.step()
                
                if (epoch + 1) % 15 == 0:
                    avg_loss = total_loss / len(train_loader)
                    print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
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
            
            config.update({
                'accuracy': accuracy,
                'f1_score': f1,
                'training_time': training_time
            })
            
            results.append(config)
            
            print(f"  Results: Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_config = config.copy()
                
                # Save best model
                model_filename = f"best_hyperopt_model_{accuracy:.4f}.pth"
                torch.save(model.state_dict(), model_filename)
                print(f"  NEW BEST MODEL SAVED: {model_filename}")
                
                # Export to ONNX
                dummy_input = torch.randn(1, input_size).to(device)
                onnx_filename = f"best_hyperopt_model_{accuracy:.4f}.onnx"
                torch.onnx.export(model, dummy_input, onnx_filename, 
                                input_names=['input'], output_names=['output'],
                                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            continue
    
    return results, best_model, best_config

def grid_search_top_configs(top_n=5, epochs=40):
    """Grid search on most promising configurations"""
    print(f"\nStarting focused grid search on top {top_n} configurations...")
    
    # Define focused search space
    focused_configs = [
        {
            'architecture': {'layers': [256, 128, 64], 'name': 'deep_256_128_64'},
            'activation': {'func': nn.ReLU(), 'name': 'relu'},
            'dropout': [0.2, 0.3],
            'batch_norm': [True, False],
            'optimizer': [
                {'name': 'Adam', 'lr': [0.001, 0.0005]},
                {'name': 'SGD', 'lr': [0.01], 'momentum': [0.9]}
            ],
            'batch_size': [32, 64],
            'weight_decay': [0.0, 0.001]
        },
        {
            'architecture': {'layers': [512, 256, 128], 'name': 'deep_512_256_128'},
            'activation': {'func': nn.LeakyReLU(0.1), 'name': 'leaky_relu'},
            'dropout': [0.2, 0.3],
            'batch_norm': [True],
            'optimizer': [
                {'name': 'Adam', 'lr': [0.001, 0.0005]}
            ],
            'batch_size': [32],
            'weight_decay': [0.001]
        }
    ]
    
    results = []
    best_accuracy = 0
    best_model = None
    best_config = None
    
    for config_template in focused_configs:
        # Generate all combinations
        arch = config_template['architecture']
        activation = config_template['activation']
        
        for dropout in config_template['dropout']:
            for batch_norm in config_template['batch_norm']:
                for batch_size in config_template['batch_size']:
                    for weight_decay in config_template['weight_decay']:
                        for opt_template in config_template['optimizer']:
                            for lr in opt_template['lr']:
                                opt_config = {'name': opt_template['name'], 'lr': lr}
                                if 'momentum' in opt_template:
                                    opt_config['momentum'] = opt_template['momentum']
                                
                                config = {
                                    'architecture': arch['name'],
                                    'layers': arch['layers'],
                                    'activation': activation['name'],
                                    'dropout': dropout,
                                    'batch_norm': batch_norm,
                                    'optimizer': opt_config['name'],
                                    'learning_rate': lr,
                                    'batch_size': batch_size,
                                    'weight_decay': weight_decay,
                                    'epochs': epochs
                                }
                                
                                print(f"\nTesting: {config}")
                                
                                try:
                                    start_time = time.time()
                                    
                                    # Create and train model
                                    model = AdvancedModel(
                                        input_size=input_size,
                                        num_classes=num_classes,
                                        layers=arch['layers'],
                                        activation=activation['func'],
                                        dropout=dropout,
                                        batch_norm=batch_norm
                                    ).to(device)
                                    
                                    optimizer = create_optimizer(model, opt_config, weight_decay)
                                    
                                    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                                    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
                                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                                    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                                    
                                    criterion = nn.CrossEntropyLoss()
                                    model.train()
                                    
                                    for epoch in range(epochs):
                                        total_loss = 0
                                        for inputs, labels in train_loader:
                                            inputs, labels = inputs.to(device), labels.to(device)
                                            
                                            optimizer.zero_grad()
                                            outputs = model(inputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer.step()
                                            
                                            total_loss += loss.item()
                                        
                                        if (epoch + 1) % 20 == 0:
                                            avg_loss = total_loss / len(train_loader)
                                            print(f"    Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
                                    
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
                                    
                                    config.update({
                                        'accuracy': accuracy,
                                        'f1_score': f1,
                                        'training_time': training_time
                                    })
                                    
                                    results.append(config)
                                    
                                    print(f"  Results: Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
                                    
                                    if accuracy > best_accuracy:
                                        best_accuracy = accuracy
                                        best_model = model
                                        best_config = config.copy()
                                        
                                        model_filename = f"best_grid_model_{accuracy:.4f}.pth"
                                        torch.save(model.state_dict(), model_filename)
                                        print(f"  NEW BEST MODEL SAVED: {model_filename}")
                                        
                                        dummy_input = torch.randn(1, input_size).to(device)
                                        onnx_filename = f"best_grid_model_{accuracy:.4f}.onnx"
                                        torch.onnx.export(model, dummy_input, onnx_filename, 
                                                        input_names=['input'], output_names=['output'],
                                                        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
                                
                                except Exception as e:
                                    print(f"  ERROR: {str(e)}")
                                    continue
    
    return results, best_model, best_config

def save_hyperopt_results(results, best_config, prefix="hyperopt"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    results_filename = f"{prefix}_results_{timestamp}.json"
    with open(results_filename, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create summary
    df_results = pd.DataFrame(results)
    summary_filename = f"{prefix}_summary_{timestamp}.csv"
    df_results.to_csv(summary_filename, index=False)
    
    print(f"\nResults saved to:")
    print(f"  Detailed: {results_filename}")
    print(f"  Summary: {summary_filename}")
    
    # Print top 10 models
    print(f"\nTop 10 Models by Accuracy:")
    sorted_results = sorted(results, key=lambda x: x['accuracy'], reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        print(f"{i:2d}. {result['architecture']} ({result['activation']}) - "
              f"Acc: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
    
    print(f"\nBest Model Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    print("Starting Advanced Hyperparameter Optimization")
    print("=" * 60)
    
    # Run random search
    random_results, random_best_model, random_best_config = random_search(num_trials=50, epochs=25)
    save_hyperopt_results(random_results, random_best_config, "random_search")
    
    # Run focused grid search
    grid_results, grid_best_model, grid_best_config = grid_search_top_configs(epochs=35)
    save_hyperopt_results(grid_results, grid_best_config, "grid_search")
    
    # Compare results
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"Random Search Best: {random_best_config['accuracy']:.4f}")
    print(f"Grid Search Best: {grid_best_config['accuracy']:.4f}")
    
    if grid_best_config['accuracy'] > random_best_config['accuracy']:
        print(f"\nOverall Best Model: Grid Search")
        print(f"Best Configuration: {grid_best_config}")
    else:
        print(f"\nOverall Best Model: Random Search")
        print(f"Best Configuration: {random_best_config}")
    
    print("\nHyperparameter optimization completed!")
