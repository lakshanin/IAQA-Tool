import os
import json
import glob
import re
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import requests

def load_sensor_data():
    print("\n--- LOADING SENSOR DATA ---")
    sensor_files = glob.glob(os.path.join("sensor-data", "*.ndjson"))
    if not sensor_files:
        print("WARNING: No sensor data files found in 'sensor-data' directory")
        return {}
    print(f"Found {len(sensor_files)} sensor data files")
    all_data = {}
    field_mappings = {}
    co2_pattern = re.compile(r'co2|carbon|ppm', re.IGNORECASE)
    temp_pattern = re.compile(r'temp|\u00b0c|celsius', re.IGNORECASE)
    humidity_pattern = re.compile(r'humid|rh|%', re.IGNORECASE)
    for file_path in sensor_files:
        try:
            room_name = os.path.basename(file_path).replace("sensor_data_", "").replace(".ndjson", "")
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
                sample_record = json.loads(first_line)
            mapping = {'timestamp': 'timestamp'}
            for field in sample_record:
                if field == 'timestamp':
                    continue
                elif co2_pattern.search(field):
                    mapping[field] = 'co2'
                elif temp_pattern.search(field):
                    mapping[field] = 'temperature'
                elif humidity_pattern.search(field):
                    mapping[field] = 'humidity'
            field_mappings[room_name] = mapping
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        normalized_record = {'room': room_name}
                        for orig_field, std_field in mapping.items():
                            normalized_record[std_field] = record[orig_field]
                        normalized_record['timestamp'] = datetime.fromisoformat(
                            normalized_record['timestamp'].replace('Z', '+00:00')
                        )
                        data.append(normalized_record)
            all_data[room_name] = data
            print(f"  Loaded {len(data)} records for {room_name}")
        except Exception as e:
            print(f"ERROR loading {file_path}: {str(e)}")
    df_data = []
    for room, data_list in all_data.items():
        df_data.extend(data_list)
    if df_data:
        df = pd.DataFrame(df_data)
        print(f"✓ Successfully loaded data with shape: {df.shape}")
        return {'dataframe': df, 'field_mappings': field_mappings, 'room_data': all_data}
    else:
        print("ERROR: No data was loaded")
        return {}

def get_code_from_llm(query, data_info, HF_TOKEN):
    print("\n--- QUERYING LLM ---")
    print(f"Processing query: '{query}'")
    df = data_info['dataframe']
    df_sample = df.head(5).to_string()
    data_structure = {
        "dataframe_info": {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in zip(df.dtypes.index, df.dtypes.values)}
        },
        "field_mappings": data_info['field_mappings']
    }
    prompt = f"""You are an expert Python data analyst. I have air quality sensor data from multiple rooms.
I need you to write Python code to answer this question:

QUESTION: {query}

The data is stored in a pandas DataFrame called 'df' with this structure:
{json.dumps(data_structure, indent=2)}

Here's a sample of the data:
{df_sample}

Important notes:
1. The data comes from different rooms with inconsistent field names that have already been standardized
2. The code should return:
   - A text summary of the findings
   - If appropriate, a pandas DataFrame with the results formatted as an HTML table
3. Your code will be executed in a context where 'df' is already available
4. Include comments in your code explaining the key steps
5. DO NOT include visualization code
6. DO NOT include code to load or preprocess the data - it's already loaded
7. The current date is {datetime.now().strftime('%Y-%m-%d')}

Return ONLY executable Python code without any explanation outside the code block. The code should:
1. Analyze the data to answer the question
2. At the end, assign a dictionary with keys 'summary' (text) and 'table_html' (HTML string from DataFrame or None) to a variable named 'result'

IMPORTANT: Your code MUST end with an assignment to a variable named 'result', like this:
result = {{
    'summary': 'your summary text here',
    'table_html': 'your table html here or None'
}}
"""
    print("Sending request to Hugging Face API...")
    try:
        url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {HF_TOKEN}"
        }
        payload = {
            "model": "moonshotai/Kimi-K2-Instruct:novita",
            "messages": [
                {"role": "system", "content": "You are an expert Python data analyst who writes code to analyze sensor data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        llm_response = response.json()
        print("✓ Received response from LLM")
        code = llm_response['choices'][0]['message']['content']
        code = re.sub(r'^```python\s*', '', code)
        code = re.sub(r'\s*```$', '', code)
        print(f"Generated code:\n{'-'*40}\n{code}\n{'-'*40}")
        return code
    except Exception as e:
        print(f"ERROR getting code from LLM: {str(e)}")
        return None

def execute_code(code, data_info):
    print("\n--- EXECUTING GENERATED CODE ---")
    namespace = {
        'df': data_info['dataframe'],
        'pd': pd,
        'np': np,
        'datetime': datetime
    }
    try:
        print("Executing code...")
        last_expression = code.strip().split('\n')[-1].strip()
        if last_expression.startswith('{') and last_expression.endswith('}'):
            modified_code = code.strip().split('\n')
            modified_code[-1] = "result = " + last_expression
            code = '\n'.join(modified_code)
            print("Modified code to assign result variable")
        exec(code, namespace)
        if 'result' not in namespace:
            print("ERROR: Code execution did not produce a 'result' variable")
            return {
                'summary': "Error: The analysis code did not return expected results.",
                'table_html': None
            }
        result = namespace['result']
        print("✓ Code executed successfully")
        print(f"Result summary: {result.get('summary', '')[:100]}...")
        if 'table_html' in result and result['table_html'] is not None:
            print("✓ Table HTML generated")
        else:
            print("No table HTML generated")
        return result
    except Exception as e:
        print(f"ERROR executing code: {str(e)}")
        traceback_str = traceback.format_exc()
        print(f"Traceback:\n{traceback_str}")
        return {
            'summary': f"Error executing analysis: {str(e)}",
            'table_html': None,
            'error': traceback_str
        }
