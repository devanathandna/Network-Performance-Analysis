# 📡 Network Performance Analysis

A Data Science project integrating **Machine Learning**, **Cybersecurity**, and **Real-time Visualization** to monitor and optimize network performance. Developed using packet analysis tools, time-series databases, and predictive models — this solution is built to proactively identify and resolve network issues in environments like academic institutions.

> 📁 **Project Folder**: [`GITHUB`](./GITHUB)  
> 📄 **Report PDF**: [`Project Documentation`](./Data_Science_Project_Template.pdf)

---

## 🧠 Abstract

This project captures and analyzes network traffic using `TShark` and processes it into **InfluxDB**, where it's visualized in **Grafana**. A **Flask** backend enables live control and integration of:

- 📈 **Latency prediction** with **LSTM**
- 🔍 **Anomaly classification** using **Isolation Forest**
- 🛡️ **DDoS detection** using **One-Class SVM**

  ![Worflow of the Project](https://github.com/devanathandna/Network-Performance-Analysis/blob/1e96c580dd1093d5db0161943ae5d9bc1c31feab/workflow.png)

---

## 🗂️ Table of Contents

- [🧠 Abstract](#-abstract)
- [🖥️ Tech Stack](#️-tech-stack)
- [🧪 Methodology](#-methodology)
- [📊 Results](#-results)
- [🎥 Demo](#-demo)


---

## 🖥️ Tech Stack

| Category        | Technologies Used                          |
|----------------|---------------------------------------------|
| **Data Capture** | TShark (CLI for Wireshark)                |
| **Data Storage** | InfluxDB (Time-series Database)           |
| **Visualization** | Grafana Dashboard                        |
| **Backend**     | Flask (Python Web Framework)               |
| **ML Models**   | LSTM, Isolation Forest, One-Class SVM      |
| **Languages**   | Python, Bash                               |

---

## 🧪 Methodology

1. **Packet Capture**: Using `TShark` to capture live packet data.
2. **Preprocessing**: Cleaning and extracting features like latency, bandwidth, jitter, and packet loss.
3. **Model Training**:
   - **LSTM**: For latency forecasting.
   - **Isolation Forest**: Network health classification.
   - **One-Class SVM**: For anomaly detection & DDoS alerts.
4. **Visualization**: Real-time Grafana dashboards embedded in Flask.
5. **User Controls**: Start/Stop capture and live updates through the web interface.

![Packet Collection Flow](https://github.com/devanathandna/Network-Performance-Analysis/blob/1e96c580dd1093d5db0161943ae5d9bc1c31feab/packet_collection.png)

---

## 📊 Results

- 📌 Real-time performance metrics (latency, packet loss, etc.)
- 📉 Predictive latency trend graphs
- 🟥 Alerts on anomaly detection (e.g., potential DDoS attacks)
- 📋 Dashboard embedded in Flask server

---

## 🎥 Demo

https://github.com/user-attachments/assets/a4a9d5d8-446c-4297-a1c6-b8af914b8f97

