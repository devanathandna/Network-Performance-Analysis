import os
import pandas as pd
from influxdb_client import InfluxDBClient
from datetime import datetime

# InfluxDB Configuration
os.environ['INFLUXDB_TOKEN'] = 'r-_t481dCecN28vh0AJ7UcgfbxuxLKNzJFlmxXSqtXbvuobcXqSSjYzzdYFACUjWYcHr8TKFKTfLlf1M97KbSw=='  # Make sure your token is properly set
token = os.environ.get("INFLUXDB_TOKEN")
org = "self"
url = "http://localhost:8086"
bucket = "Network_Performance"

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

def parse_and_write_influx_to_csv(output_file="captured_data.csv"):
    """
    Fetch data from InfluxDB and write it to a CSV file.
    """
    print("Starting data parsing and writing to CSV.")
    influx_data = fetch_from_influxdb()
    if influx_data:
        df = process_data_to_dataframe(influx_data)
        write_to_csv(df, file_name=output_file)
    else:
        print("No data fetched from InfluxDB.")

def fetch_from_influxdb():
    """
    Fetch data from InfluxDB.
    Modify the query as needed to fetch the relevant data.
    """
    query = '''
    from(bucket: "Network_Performance")
    |> range(start: -1m)  // Adjust the time range as needed
    |> filter(fn: (r) => r["_measurement"] == "network_metrics")
    |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    |> sort(columns: ["_time"], desc: false)
    '''
    try:
        result = query_api.query(query=query)
        return result
    except Exception as e:
        print(f"Error fetching data from InfluxDB: {e}")
        return None

def process_data_to_dataframe(data):
    """
    Convert InfluxDB data to pandas DataFrame.
    """
    try:
        rows = []
        for table in data:
            for record in table.records:
                rows.append({
                    "Time": record.get_time(),
                    **record.values
                })

        df = pd.DataFrame(rows)
        return df
    except Exception as e:
        print(f"Error processing data to DataFrame: {e}")
        return None

def write_to_csv(df, file_name="captured_data.csv"):
    """
    Write the pandas DataFrame to a CSV file.
    """
    try:
        if df is not None and not df.empty:
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
            print(f"Data successfully written to {file_name}.")
        else:
            print("No data available to write to CSV.")
    except Exception as e:
        print(f"Error writing data to CSV: {e}")

# Closing the client after usage
def close_client():
    """
    Ensure the InfluxDB client is properly closed after use.
    """
    client.close()
    print("InfluxDB client closed.")
