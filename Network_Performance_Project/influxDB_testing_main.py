import os
import pandas as pd
import time
from influxdb_client import InfluxDBClient
from datetime import datetime

# InfluxDB Configuration
os.environ['INFLUXDB_TOKEN'] = 'r-_t481dCecN28vh0AJ7UcgfbxuxLKNzJFlmxXSqtXbvuobcXqSSjYzzdYFACUjWYcHr8TKFKTfLlf1M97KbSw=='
token = os.environ.get("INFLUXDB_TOKEN")
org = "self"
url = "http://localhost:8086"
bucket = "Network_Performance"

# Initialize InfluxDB client
client = InfluxDBClient(url=url, token=token, org=org)
query_api = client.query_api()

def fetch_from_influxdb():
    """
    Fetch data from InfluxDB within a specific time range.
    Modify the query as per requirements to filter data.
    """
    query = f'''
    from(bucket: "{bucket}")
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
    Process the fetched InfluxDB data into a pandas DataFrame.
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

def write_to_csv(df, file_name="network_data_collection.csv"):
    """
    Write the DataFrame data to a CSV file.
    """
    try:
        if df is not None and not df.empty:
            df.to_csv(file_name, mode='a', header=not os.path.exists(file_name), index=False)
            print(f"Data successfully appended to {file_name}.")
        else:
            print("No data available to write to CSV.")
    except Exception as e:
        print(f"Error writing data to CSV: {e}")

if __name__ == "__main__":
    try:
        print("Press CTRL+C to stop capturing data.")
        while True:
            # Fetch data from InfluxDB
            influx_data = fetch_from_influxdb()

            if influx_data:
                # Process data into a DataFrame
                df = process_data_to_dataframe(influx_data)

                if df is not None:
                    # Print the data to the terminal
                    print("Fetched Data:")
                    print(df)

                    # Append the DataFrame to the CSV
                    write_to_csv(df, file_name="network_data_collection.csv")

          
            time.sleep(2) 
    except KeyboardInterrupt:
        print("\nData capture stopped by user.")
    finally:
        client.close()
        print("InfluxDB client closed.")
