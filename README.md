# Air Quality Sensor Analyzer

A full-stack AI Agent web app for analyzing air quality sensor data using natural language queries.

## Setup

1. Make sure you have Python 3.8+ installed
2. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Ensure you have a `.env` file with your Hugging Face API token:
   ```
   HF_TOKEN=your_huggingface_token
   ```
4. Make sure your sensor data is in the `sensor-data` folder

## Usage

Run the application with a single command:

```
python app.py
```

Then open your browser to http://localhost:5000

## Features

- Natural language queries to analyze sensor data
- Automatic handling of inconsistent field names across files
- Intelligent code generation for data analysis
- Results displayed as both text summaries and tables
- Simple and responsive UI

## Sample Queries

- How does CO2 vary by day of week?
- Which room had highest temperature last week?
- List rooms by average humidity
- Show temperature changes hourly in Room 1
