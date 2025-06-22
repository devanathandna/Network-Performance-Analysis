import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA

# Define your LSTM model here if it's not in another module
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.4):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.bn = nn.BatchNorm1d(2 * hidden_dim)  # Batch normalization after LSTM
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take the last output (from both directions)
        out = self.bn(out)  # Apply batch normalization
        out = self.fc(out)  # Pass through fully connected layer
        return out

def load_models():
    try:
        iso_model = joblib.load('models/isolation_forest_model_revised.pkl')
        svm_model = joblib.load('models/one_class_svm_model_revised.pkl')
        return iso_model, svm_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

def train_isolation_forest_with_three_conditions(data, contamination=0.01):
    try:
        model = IsolationForest(contamination=contamination, random_state=42)
        model.fit(data[['PC1', 'PC2']])
        labels = model.predict(data[['PC1', 'PC2']])
        data['network_condition'] = 'Normal'
        data.loc[labels == -1, 'network_condition'] = 'Critical'
        return model, data
    except Exception as e:
        print(f"Error training Isolation Forest: {e}")
        return None, data

def train_one_class_svm_for_ddos(data, nu=0.001, kernel='rbf'):
    try:
        model = OneClassSVM(nu=nu, kernel=kernel, gamma='scale')
        model.fit(data[['PC1', 'PC2']])
        labels = model.predict(data[['PC1', 'PC2']])
        data['condition'] = np.where(labels == 1, 'Normal', 'Anomaly')
        return model, data
    except Exception as e:
        print(f"Error training One-Class SVM: {e}")
        return None, data

def process_data(file_path):
    try:
        # Load the data
        data = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    try:
        # Data preprocessing steps
        
        data.columns = data.columns.str.strip()


        data = data.dropna(subset=['src_ip', 'dst_ip'])
        data['ack_rtt_ms'] = pd.to_numeric(data['ack_rtt_ms'], errors='coerce')
        data['ttl'] = pd.to_numeric(data['ttl'], errors='coerce')
        data['tcp_src_port'] = pd.to_numeric(data['tcp_src_port'], errors='coerce')
        data['tcp_dst_port'] = pd.to_numeric(data['tcp_dst_port'], errors='coerce')
        data['window_size'] = pd.to_numeric(data['window_size'], errors='coerce')
        data['time_delta_ms'] = pd.to_numeric(data['time_delta_ms'], errors='coerce')
        data['packet_length_bytes'] = pd.to_numeric(data['packet_length_bytes'], errors='coerce')
        data['retransmission'] = pd.to_numeric(data['retransmission'], errors='coerce')
        data['retransmission'] = data['retransmission'].notnull().astype(int)

        # Impute missing values
        mode_imputer = SimpleImputer(strategy='most_frequent')
        data[['jitter_ms', 'tcp_flags']] = mode_imputer.fit_transform(data[['jitter_ms', 'tcp_flags']])

        # Select numerical features for PCA
        features = ['ack_rtt_ms', 'avg_latency_ms', 'bandwidth_mbps', 'jitter_ms', 
                    'packet_length_bytes', 'retransmission', 'session_duration_sec', 
                    'tcp_dst_port', 'tcp_src_port', 'time_delta_ms', 'total_data_mb', 
                    'ttl', 'window_size', 'tcp_flags']
        
        data = data[features].dropna()
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
        
        iso_model, pca_df = train_isolation_forest_with_three_conditions(pca_df)
        pca_df['isolation_forest_prediction'] = iso_model.predict(pca_df[['PC1', 'PC2']])
        # Anomaly detection using One-Class SVM
        svm_model, pca_df = train_one_class_svm_for_ddos(pca_df)
        pca_df['svm_prediction'] = svm_model.predict(pca_df[['PC1', 'PC2']])
        
        
        # Initialize and load the LSTM model
        model = LSTMModel(input_dim=2, hidden_dim=64, num_layers=3, output_dim=1)  # Adjust parameters as needed
        model.load_state_dict(torch.load('models/lstm_model_revised_main.pth'))  # Load the state dict
        model.eval()
        
        X_lstm = pca_df[['PC1', 'PC2']].values
        X_lstm = torch.tensor(X_lstm, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        # Make predictions
        with torch.no_grad():
            predictions = model(X_lstm)  # Forward pass
            predictions = predictions.squeeze().numpy()  # Remove batch dimension

        # Add predictions to the DataFrame
        pca_df['predicted_avg_latency_ms'] = predictions
        
        return pca_df 
    except Exception as e:
        print(f"Error processing data: {e}")
        return None