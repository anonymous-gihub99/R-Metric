# enhanced_baseline_comparison.py
"""
Enhanced baseline comparison with fixed Isolation Forest and new LSTM Autoencoder
Built on your existing baseline_comparison.py structure
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import warnings
warnings.filterwarnings('ignore')


class BaselineMethod:
    """Base class for baseline methods"""
    
    def __init__(self, name: str):
        self.name = name
        self.alerts = []
        self.scores = []
        self.first_alert_step = None
        
    def update(self, step: int, metrics: Dict) -> bool:
        """Update method and return alert status"""
        raise NotImplementedError
        
    def reset(self):
        """Reset method state"""
        self.alerts = []
        self.scores = []
        self.first_alert_step = None


class FixedIsolationForestBaseline(BaselineMethod):
    """Fixed Isolation Forest with proper feature engineering and tuning"""
    
    def __init__(self, contamination: float = 0.05, window_size: int = 30):
        super().__init__("Fixed Isolation Forest")
        self.contamination = contamination
        self.window_size = window_size
        self.feature_history = []
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=150,  # Increased for better performance
            max_samples='auto',
            max_features=1.0,
            bootstrap=True,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.training_phase = True
        self.training_samples = 100  # Need samples to train properly
        
    def extract_features(self, metrics: Dict, history: List[Dict]) -> np.ndarray:
        """Extract comprehensive features for anomaly detection"""
        features = []
        
        # Current metrics
        current_loss = metrics.get('train_loss', 0)
        features.append(current_loss)
        
        # Gradient norm (important signal)
        features.append(metrics.get('grad_norm', np.random.normal(10, 2)))
        
        # Loss statistics over window
        if len(history) >= 5:
            recent_losses = [h.get('train_loss', current_loss) for h in history[-5:]]
            features.append(np.mean(recent_losses))
            features.append(np.std(recent_losses))
            features.append(np.max(recent_losses) - np.min(recent_losses))
            
            # Rate of change
            if len(history) >= 10:
                older_losses = [h.get('train_loss', current_loss) for h in history[-10:-5]]
                features.append(np.mean(recent_losses) - np.mean(older_losses))
            else:
                features.append(0)
        else:
            features.extend([current_loss, 0, 0, 0])
        
        # Learning rate (if available)
        features.append(metrics.get('learning_rate', 5e-5))
        
        # Add R-metric components if available
        features.append(metrics.get('delta_l', 0))
        features.append(metrics.get('sigma_sq', 0))
        
        return np.array(features)
    
    def update(self, step: int, metrics: Dict) -> bool:
        """Update with improved logic"""
        # Store metrics history
        self.feature_history.append(metrics)
        
        # Extract features
        features = self.extract_features(metrics, self.feature_history)
        
        # Need minimum samples to start
        if len(self.feature_history) < 50:
            self.alerts.append(False)
            self.scores.append(0.0)
            return False
        
        # Prepare training data
        X_train = []
        for i in range(max(0, len(self.feature_history) - self.window_size), len(self.feature_history)):
            X_train.append(self.extract_features(
                self.feature_history[i], 
                self.feature_history[:i] if i > 0 else []
            ))
        
        if len(X_train) < 10:
            self.alerts.append(False)
            self.scores.append(0.0)
            return False
        
        X_train = np.array(X_train)
        
        # Fit or update model
        if not self.is_fitted or step % 50 == 0:
            X_scaled = self.scaler.fit_transform(X_train[:-1])
            self.model.fit(X_scaled)
            self.is_fitted = True
        
        # Predict on current
        current_scaled = self.scaler.transform(features.reshape(1, -1))
        prediction = self.model.predict(current_scaled)[0]
        anomaly_score = -self.model.score_samples(current_scaled)[0]
        
        # Normalize score to [0, 1]
        normalized_score = 1 / (1 + np.exp(-2 * anomaly_score))
        
        alert = prediction == -1 and normalized_score > 0.7  # More conservative threshold
        
        self.alerts.append(alert)
        self.scores.append(normalized_score)
        
        if alert and self.first_alert_step is None:
            self.first_alert_step = step
        
        return alert


class LSTMAutoencoderBaseline(BaselineMethod):
    """LSTM Autoencoder for anomaly detection"""
    
    def __init__(self, sequence_length: int = 20, hidden_size: int = 64, threshold_std: float = 2.5):
        super().__init__("LSTM Autoencoder")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.threshold_std = threshold_std
        self.model = self._build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.loss_history = deque(maxlen=100)
        self.reconstruction_errors = deque(maxlen=50)
        self.is_trained = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _build_model(self):
        """Build LSTM Autoencoder model"""
        class LSTMAutoencoder(nn.Module):
            def __init__(self, input_size=1, hidden_size=64, num_layers=2):
                super().__init__()
                # Encoder
                self.encoder = nn.LSTM(
                    input_size, hidden_size, num_layers, 
                    batch_first=True, dropout=0.1
                )
                # Decoder
                self.decoder = nn.LSTM(
                    hidden_size, hidden_size, num_layers, 
                    batch_first=True, dropout=0.1
                )
                self.output_layer = nn.Linear(hidden_size, input_size)
                
            def forward(self, x):
                # Encode
                encoded, (hidden, cell) = self.encoder(x)
                
                # Decode
                decoded, _ = self.decoder(encoded, (hidden, cell))
                output = self.output_layer(decoded)
                
                return output
        
        return LSTMAutoencoder()
    
    def create_sequences(self, data: List[float]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create sequences for training"""
        if len(data) < self.sequence_length:
            return None, None
        
        sequences = []
        for i in range(len(data) - self.sequence_length):
            seq = data[i:i + self.sequence_length]
            sequences.append(seq)
        
        if not sequences:
            return None, None
        
        sequences = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1)
        return sequences, sequences  # For autoencoder, input = target
    
    def train_model(self, sequences: torch.Tensor, epochs: int = 5):
        """Quick training of the autoencoder"""
        if sequences is None or len(sequences) < 5:
            return
        
        sequences = sequences.to(self.device)
        self.model.train()
        
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = self.loss_fn(outputs, sequences)
            loss.backward()
            self.optimizer.step()
        
        self.is_trained = True
    
    def update(self, step: int, metrics: Dict) -> bool:
        """Update LSTM autoencoder baseline"""
        current_loss = metrics.get('train_loss', 0)
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < self.sequence_length + 10:
            self.alerts.append(False)
            self.scores.append(0.0)
            return False
        
        # Create sequences
        sequences, _ = self.create_sequences(list(self.loss_history))
        
        if sequences is None or len(sequences) < 5:
            self.alerts.append(False)
            self.scores.append(0.0)
            return False
        
        # Train or update model periodically
        if not self.is_trained or step % 100 == 0:
            train_sequences = sequences[:-1]  # Keep last for testing
            if len(train_sequences) > 0:
                self.train_model(train_sequences, epochs=10)
        
        if not self.is_trained:
            self.alerts.append(False)
            self.scores.append(0.0)
            return False
        
        # Get reconstruction error for current sequence
        self.model.eval()
        with torch.no_grad():
            test_seq = sequences[-1:].to(self.device)
            reconstruction = self.model(test_seq)
            error = self.loss_fn(reconstruction, test_seq).item()
        
        self.reconstruction_errors.append(error)
        
        # Calculate anomaly score
        if len(self.reconstruction_errors) > 10:
            mean_error = np.mean(list(self.reconstruction_errors)[:-1])
            std_error = np.std(list(self.reconstruction_errors)[:-1])
            
            if std_error > 0:
                z_score = (error - mean_error) / std_error
                alert = z_score > self.threshold_std
                score = min(1.0, max(0.0, z_score / (2 * self.threshold_std)))
            else:
                alert = False
                score = 0.0
        else:
            alert = False
            score = 0.0
        
        self.alerts.append(alert)
        self.scores.append(score)
        
        if alert and self.first_alert_step is None:
            self.first_alert_step = step
        
        return alert


class EnhancedBaselineComparison:
    """Enhanced comparison with all baselines"""
    
    def __init__(self, config_path: str = None):
        self.methods = self._initialize_methods()
        self.results = defaultdict(list)
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            return {
                'fault_injection_step': 400,
                'max_steps': 800,
                'r_metric_alert_threshold': 0.6
            }
    
    def _initialize_methods(self) -> List[BaselineMethod]:
        """Initialize all baseline methods"""
        from baseline_comparison import (
            SimpleHeuristicBaseline,
            LossSpikeBaseline,
            GradientNormBaseline,
            MovingAverageBaseline
        )
        
        methods = [
            SimpleHeuristicBaseline(threshold=3),
            LossSpikeBaseline(window_size=20, z_threshold=2.5),
            GradientNormBaseline(threshold=50.0),
            FixedIsolationForestBaseline(contamination=0.05),
            LSTMAutoencoderBaseline(sequence_length=20),
            MovingAverageBaseline(window_size=20, sensitivity=2.0)
        ]
        
        return methods
    
    def run_comparison(self, metrics_df: pd.DataFrame) -> Dict:
        """Run all methods on the data"""
        # Reset all methods
        for method in self.methods:
            method.reset()
        
        # Process each step
        for idx, row in metrics_df.iterrows():
            step = row['step']
            metrics = row.to_dict()
            
            # Update each method
            for method in self.methods:
                try:
                    alert = method.update(step, metrics)
                except Exception as e:
                    print(f"Error in {method.name}: {e}")
                    alert = False
        
        # Compile results
        results = {}
        fault_step = self.config['fault_injection_step']
        
        for method in self.methods:
            # Find first alert
            alerts = np.array(method.alerts)
            alert_indices = np.where(alerts)[0]
            
            if len(alert_indices) > 0:
                first_alert_idx = alert_indices[0]
                first_alert_step = metrics_df.iloc[min(first_alert_idx, len(metrics_df)-1)]['step']
                lead_time = fault_step - first_alert_step
            else:
                first_alert_step = None
                lead_time = None
            
            # Calculate metrics
            tp = sum(alerts[metrics_df['step'] >= fault_step]) if any(alerts) else 0
            fp = sum(alerts[metrics_df['step'] < fault_step]) if any(alerts) else 0
            
            # Should detect after fault
            should_detect = len(metrics_df[metrics_df['step'] >= fault_step])
            fn = should_detect - tp
            tn = len(metrics_df[metrics_df['step'] < fault_step]) - fp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[method.name] = {
                'first_alert_step': first_alert_step,
                'lead_time': lead_time,
                'total_alerts': sum(alerts),
                'alert_rate': np.mean(alerts),
                'scores': method.scores,
                'detected': first_alert_step is not None,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn
            }
        
        return results