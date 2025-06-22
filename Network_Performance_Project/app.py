from flask import Flask, render_template, jsonify
import subprocess
import os
import threading
import ipaddress
from threading import Thread
import numpy as np
import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
from influx_testing_main import parse_and_write_influx_to_csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from data_precessing import process_data
import time

# Set pandas option
pd.set_option('future.no_silent_downcasting', True)

# Flask App Initialization
app = Flask(__name__)

# InfluxDB Configuration
os.environ['INFLUXDB_TOKEN'] = 'r-_t481dCecN28vh0AJ7UcgfbxuxLKNzJFlmxXSqtXbvuobcXqSSjYzzdYFACUjWYcHr8TKFKTfLlf1M97KbSw=='
token = os.environ.get("INFLUXDB_TOKEN")
org = "self"
url = "http://localhost:8086"

client = InfluxDBClient(url=url, token=token, org=org)
bucket = "Network_Performance"
write_api = client.write_api(write_options=SYNCHRONOUS)

# ------------------------------------------------------------
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)  # Show all rows (optional)
pd.set_option('display.width', 1000)  # Set the width to avoid line breaks
pd.set_option('display.colheader_justify', 'center') 
#-------------------------------------------------------------

# Utility Function
def ip_to_int(ip):
    try:
        return int(ipaddress.ip_address(ip))
    except ValueError:
        print(f"Invalid IP address: {ip}")
        return 0

# Data Columns
columns = [
    "time", "src_ip", "dst_ip", "protocol", "packet_length", "tcp_src_port",
    "tcp_dst_port", "ttl", "tcp_flags", "window_size", "ack_rtt",
    "retransmission", "time_delta"
]
data = pd.DataFrame(columns=columns)

# TShark Command with explicit field ordering
tshark_command = [
    "tshark", "-i", "Wi-Fi", "-T", "fields",
    "-E", "header=y", "-E", "separator=,",
    "-e", "frame.time_epoch",     # Field 0: time
    "-e", "ip.src",               # Field 1: source IP
    "-e", "ip.dst",               # Field 2: destination IP
    "-e", "_ws.col.Protocol",     # Field 3: protocol
    "-e", "frame.len",            # Field 4: packet length
    "-e", "tcp.srcport",          # Field 5: TCP source port
    "-e", "tcp.dstport",          # Field 6: TCP destination port
    "-e", "ip.ttl",               # Field 7: TTL
    "-e", "tcp.flags",            # Field 8: TCP flags
    "-e", "tcp.window_size_value",# Field 9: window size
    "-e", "tcp.analysis.ack_rtt", # Field 10: ACK RTT
    "-e", "tcp.analysis.retransmission", # Field 11: retransmission
    "-e", "frame.time_delta"      # Field 12: time delta
]

# Global Variables
tshark_process = None
capturing = False

# Capture Function
def start_packet_capture():
    global tshark_process, capturing, data
    capturing = True

    process = subprocess.Popen(tshark_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        while capturing:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break

            if output:
                row = [item.strip() for item in output.strip().split(',')]
                if len(row) < 13:
                    print(f"Skipping malformed row (insufficient fields): {row}")
                    continue

                try:
                # Validate and convert IPs
                    src_ip = ip_to_int(row[1])
                    dst_ip = ip_to_int(row[2])
                    
                    if np.isnan(src_ip) or np.isnan(dst_ip):
                        print(f"Skipping row due to invalid IP addresses: {row}")
                        continue
                    
                    
                    try:
                        packet_length = float(row[4]) if row[4] else np.nan
                    except ValueError:
                        print(f"Invalid packet length value: {row[4]}")
                        continue
                    
                    
                    
                    data_row = {
                        "time": float(row[0]) if row[0] else np.nan,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "protocol": row[3] if row[3] else "UNKNOWN",
                        "packet_length": packet_length*1000,
                        "tcp_src_port": float(row[5]) if row[5] else np.nan,
                        "tcp_dst_port": float(row[6]) if row[6]else np.nan,
                        "ttl": float(row[7]) if row[7] else np.nan,
                        "tcp_flags": float(int(row[8], 16)) if row[8] and row[8].startswith('0x') else (float(row[8]) if row[8] else np.nan),
                        "window_size": float(row[9]) if row[9] else np.nan,
                        "ack_rtt": float(row[10]) if row[10] else np.nan,
                        "retransmission": int(row[11]) if row[11] else 0,
                        "time_delta": float(row[12]) if row[12] else np.nan
                    }
                    
                    
                    
                    data = pd.concat([data, pd.DataFrame([data_row])], ignore_index=True)
                    #print(f"Summed total data before grouping: {data['packet_length'].sum()}")
                    
                    
                    # Process data when buffer reaches 1000 rows
                    if len(data) >= 1000:
                        print("Processing batch of 1000 records...")
                        
                        # Imputation
                        median_imputer = SimpleImputer(strategy='median')
                        mode_imputer = SimpleImputer(strategy='most_frequent')
                        data[['ack_rtt', 'ttl']] = median_imputer.fit_transform(data[['ack_rtt', 'ttl']])
                        data[['tcp_src_port', 'tcp_dst_port', 'window_size']] = mode_imputer.fit_transform(
                            data[['tcp_src_port', 'tcp_dst_port', 'window_size']]
                        )
                        
                        # Scaling
                        scaler = MinMaxScaler()
                        corr_features2 = ['packet_length', 'tcp_flags', 'ack_rtt', 'window_size', 'time_delta', 'ttl']
                        data[corr_features2] = scaler.fit_transform(data[corr_features2])
                        
                        # Calculate metrics
                        avg_latency = data.groupby(['src_ip', 'dst_ip'])['time_delta'].mean().rename("avg_latency")
                        jitter = data.groupby(['src_ip', 'dst_ip'])['time_delta'].std().rename("jitter")
                        total_data = data.groupby(['src_ip', 'dst_ip'])['packet_length'].sum().rename("total_data")
                        session_duration = data.groupby(['src_ip', 'dst_ip'])['time'].apply(
                            lambda x: x.max() - x.min()).rename("session_duration")

                        # Avoid division by zero
                        session_duration = session_duration.replace({0: 0.001})

                        # Calculate bandwidth (in Mbps)
                        bandwidth = ((total_data * 8) / session_duration / 1e6).fillna(0.).rename("bandwidth")

                        
                        additional_columns = ['protocol', 'packet_length', 'tcp_src_port', 'tcp_dst_port', 
                                            'ttl', 'tcp_flags', 'window_size', 'ack_rtt', 'retransmission', 'time_delta']
                        representative_data = data.groupby(['src_ip', 'dst_ip']).first()[additional_columns]

                        # Combine Metrics
                        metrics_df = pd.concat([avg_latency, jitter, total_data, session_duration, bandwidth, representative_data], axis=1).reset_index()
                        metrics_df = metrics_df.dropna(subset=["bandwidth"])

                        
                        metrics_df['avg_latency_ms'] = (metrics_df['avg_latency'] * 1000).round(2)  # Convert seconds to ms
                        metrics_df['jitter_ms'] = (metrics_df['jitter'] * 1000).round(2)            # Convert seconds to ms
                        metrics_df['total_data_mb'] = (metrics_df['total_data']*10).round(2)  # Convert bytes to MB
                        metrics_df['session_duration_sec'] = metrics_df['session_duration'].round(2)      # Seconds
                        metrics_df['bandwidth_mbps'] = metrics_df['bandwidth'].round(2)                   # Mbps
                        metrics_df['packet_length_bytes'] = metrics_df['packet_length'].round(2)          # Bytes
                        metrics_df['ack_rtt_ms'] = (metrics_df['ack_rtt'] * 1000).round(2)                # Convert seconds to ms
                        metrics_df['time_delta_ms'] = (metrics_df['time_delta'] * 1000).round(2)          # Convert seconds to ms

                        # Write to InfluxDB
                        print("Writing to InfluxDB...")
                        for _, row in metrics_df.iterrows():
                            point = (
                                Point("network_metrics")
                                .tag("src_ip", row["src_ip"])
                                .tag("dst_ip", row["dst_ip"])
                                .tag("protocol", row["protocol"])
                                .field("avg_latency_ms", row["avg_latency_ms"])
                                .field("jitter_ms", row["jitter_ms"])
                                .field ("total_data_mb", row["total_data_mb"])
                                .field("session_duration_sec", row["session_duration_sec"])
                                .field("bandwidth_mbps", row["bandwidth_mbps"])
                                .field("packet_length_bytes", row["packet_length_bytes"])
                                .field("tcp_src_port", row["tcp_src_port"])
                                .field("tcp_dst_port", row["tcp_dst_port"])
                                .field("ttl", row["ttl"])
                                .field("tcp_flags", row["tcp_flags"])
                                .field("window_size", row["window_size"])
                                .field("ack_rtt_ms", row["ack_rtt_ms"])
                                .field("retransmission", row["retransmission"])
                                .field("time_delta_ms", row["time_delta_ms"])
                            )
                            write_api.write(bucket=bucket, org=org, record=point)
                            
                            print(f"The Dataframe of metric_df is {metrics_df.head(5)}")
                        
                        # Reset data buffer
                        
                        data = pd.DataFrame(columns=columns)
                        print(row)
                        print("Batch processing complete. Buffer reset.")
                    
                except ValueError as e:
                    print(f"Error processing row: {row}")
                    print(f"Error details: {e}")
                    continue
                except Exception as e:
                    print(f"Unexpected error: {e}")
                    print(f"Row causing error: {row}")
                    continue

    except KeyboardInterrupt:
        print("\nData collection stopped by user.")
    finally:
        process.terminate()
        print("TShark process terminated.")
        client.close()
        print("InfluxDB connection closed.")
        
def csv_write_background():
    print("Starting CSV write process.")
    
    try:
        source_file = "captured_data.csv"
        destination_file = "new_data.csv"  

        if os.path.exists(source_file):
            df = pd.read_csv(source_file)
            df.to_csv(destination_file, index=False)
            print(f"Data successfully written to {destination_file}.")
        else:
            print("Source CSV file does not exist.")
    except Exception as e:
        print(f"Error writing to new CSV: {str(e)}")

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global capturing
    if not capturing:
        threading.Thread(target=start_packet_capture).start()
    return jsonify({"status": "capturing started"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global capturing, tshark_process
    capturing = False
    if tshark_process:
        tshark_process.terminate()
        tshark_process = None
    return jsonify({"status": "capturing stopped"})


@app.route('/csv_write', methods=['POST'])
def csv_write():
    thread = Thread(target=csv_write_background)
    thread.start()
    return jsonify({"status": "CSV write process started."})


@app.route('/get_predictions', methods=['GET'])
def get_predictions():
    try:
        
        csv_file_path = "captured_data.csv"
        if os.path.exists(csv_file_path):
            os.remove(csv_file_path)
            print(f"Deleted existing CSV file: {csv_file_path}")
        
        try:
            parse_and_write_influx_to_csv(output_file=csv_file_path)
            print("Data fetched from InfluxDB and written to CSV.")
        except Exception as e:
            print(f"Error fetching data from InfluxDB: {str(e)}")
            return jsonify({"error": "Failed to fetch data from InfluxDB."}), 500

        
        time.sleep(15)  

        
        try:
            predictions = process_data(csv_file_path)
            predictions_list = predictions['predicted_avg_latency_ms'].tolist()
            isolation_forest_predictions = predictions['isolation_forest_prediction'].tolist()
            svm_predictions = predictions['svm_prediction'].tolist()
            print("Predictions processed successfully.")
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return jsonify({"error": "Failed to process predictions."}), 500

        # Return all predictions as JSON
        return jsonify({
            'latency_predictions': predictions_list,
            'isolation_forest_predictions': isolation_forest_predictions,
            'svm_predictions': svm_predictions
        })

    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred."}), 500

# Main Entry Point
if __name__ == '__main__':
    app.run(debug=True)