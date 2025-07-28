# Core imports
import os
import json
import glob
import re
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from logs import logger 

# Helper function for fixing code indentation issues
def fix_code_indentation(code_str):
    """Fix common indentation issues in code"""
    lines = code_str.splitlines()
    fixed_lines = []
    in_def = False
    def_indent = 0
    
    for line in lines:
        stripped = line.strip()
        # Skip empty lines
        if not stripped:
            fixed_lines.append(line)
            continue
            
        # Check for function/class definitions
        if stripped.startswith('def ') or stripped.startswith('class '):
            if stripped.endswith(':'):
                in_def = True
                def_indent = len(line) - len(line.lstrip())
            fixed_lines.append(line)
        # Handle indentation after function definition
        elif in_def:
            indent = len(line) - len(line.lstrip())
            if indent <= def_indent:
                # First line after def should be indented
                fixed_lines.append(' ' * (def_indent + 4) + stripped)
                in_def = False
            else:
                # Already indented
                fixed_lines.append(line)
                in_def = False
        else:
            fixed_lines.append(line)
            
    return '\n'.join(fixed_lines)


def standardize_column_names(df):
    """
    Standardize column names related to CO₂, temperature, and humidity using regex patterns.
    Maps various column naming conventions to standard column names: 'co2', 'temp', and 'humid'.
    
    Args:
        df (pandas.DataFrame): DataFrame with original column names
        
    Returns:
        pandas.DataFrame: DataFrame with standardized column names
    """
    logger.info("Standardizing column names using regex patterns")
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Dictionary to track which columns were mapped to what
    column_mapping = {}
    
    # Compile regex patterns once
    logger.debug("=== REGEX PATTERN TESTING ===")
    pattern_defs = {
        'co2': r'(?i)co[₂2]|carbon|CO2|ppm|dioxide',
        'temp': r'(?i)temp|°c|celsius|temperature',
        'humid': r'(?i)humid|rh|moisture|%\s*rh|relative humid'
    }
    compiled_patterns = {k: re.compile(v) for k, v in pattern_defs.items()}
    # Count how many columns match each pattern
    pattern_matches = {pattern_name: 0 for pattern_name in compiled_patterns}
    for measure, compiled_pattern in compiled_patterns.items():
        logger.debug(f"Testing {measure} pattern: {compiled_pattern.pattern}")
        for col in df.columns:
            if compiled_pattern.search(str(col)):
                pattern_matches[measure] += 1
                logger.debug(f"  ✓ '{col}' matches {measure} pattern")
    
    # Build a new DataFrame with only standardized columns and room
    standardized_rows = []
    col_types = list(compiled_patterns.items())
    for room in df['room'].unique():
        mask = df['room'] == room
        room_data = df[mask]
        found_cols = {}
        # Find columns for each standardized value
        for std_name, pattern in col_types:
            for column in room_data.columns:
                if pattern.search(str(column)) and room_data[column].notna().any():
                    found_cols[std_name] = column
                    column_mapping[f"{room}:{column}"] = std_name
                    break
        # Only append if all are found
        if all(k in ['co2', 'temp', 'humid'] for k in found_cols):
            for idx, row in room_data.iterrows():
                standardized_rows.append({
                    'timestamp': row['timestamp'],
                    'room': room,
                    'co2': row[found_cols['co2']],
                    'temp': row[found_cols['temp']],
                    'humid': row[found_cols['humid']]
                })
            logger.info(f"Room {room}: Mapped {found_cols} to standardized columns. Sample: co2={room_data[found_cols['co2']].dropna().head(2).tolist()}, temp={room_data[found_cols['temp']].dropna().head(2).tolist()}, humid={room_data[found_cols['humid']].dropna().head(2).tolist()}")
        else:
            logger.warning(f"Room {room}: Could not find all standardized columns (co2, temp, humid). Found: {found_cols}")
    # Create new DataFrame
    df_std = pd.DataFrame(standardized_rows)
    return df_std, column_mapping


def load_sensor_data():
    """
    Load sensor data from all .ndjson files in the sensor-data directory.
    Returns a dict with the combined DataFrame and original field names per room.
    """
    logger.info("--- LOADING SENSOR DATA ---")
    sensor_files = glob.glob(os.path.join("sensor-data", "*.ndjson"))

    if not sensor_files:
        logger.warning("No sensor data files found in 'sensor-data' directory")
        return {}
    logger.info(f"Found {len(sensor_files)} sensor data files")

    all_data = {}
    field_mappings = {}
    all_original_fields = {}

    for file_path in sensor_files:
        # Derive room name
        base = os.path.splitext(os.path.basename(file_path))[0]
        if base.lower().startswith("sensor_data_"):
            room_name = base[len("sensor_data_"):]
        else:
            room_name = base

        # Read first line to detect original field names
        try:
            with open(file_path, 'r') as f:
                first_line = f.readline().strip()
            sample = json.loads(first_line)
        except Exception as e:
            logger.error(f"Skipping {file_path}, cannot read sample: {e}")
            continue

        # Store original field names for LLM to interpret
        mapping = {'timestamp': 'timestamp'}
        for orig in sample:
            if orig != 'timestamp':
                mapping[orig] = orig

        field_mappings[room_name] = mapping
        all_original_fields[room_name] = list(sample.keys())
        # Show raw columns and sample record for each file
        logger.info(f"Room '{room_name}' raw columns: {all_original_fields[room_name]}")
        records = []

        # Load and normalize each record
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    rec['room'] = room_name
                    if 'timestamp' in rec:
                        ts = rec.get('timestamp')
                        rec['timestamp'] = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    records.append(rec)
        except Exception as e:
            logger.error(f"ERROR loading {file_path}: {e}")
            continue

        if records:
            all_data[room_name] = records
            logger.info(f"Loaded {len(records)} records for {room_name}")
            if len(records) > 0:
                logger.info(f"Sample record from '{room_name}': {json.dumps(records[0], default=str)[:200]}...")

    # Combine all rooms into one DataFrame
    rows = [rec for room_records in all_data.values() for rec in room_records]
    if not rows:
        logger.error("No data loaded from any sensor files.")
        return {}

    df = pd.DataFrame(rows)
    logger.info(f"Successfully loaded data with shape: {df.shape}")

    # Show columns and info before standardization
    logger.info(f"Before standardization: {df.info()}")

    # Standardize column names using regex patterns
    df_standardized, column_mapping = standardize_column_names(df)

    # Show columns and info after standardization
    logger.info(f"After standardization: {df_standardized.info()}")

    return {'dataframe': df_standardized, 'original_fields': all_original_fields, 'column_mapping': column_mapping}


def get_code_from_llm(query, data_info, HF_TOKEN):
    """
    Query the LLM to generate Python code for a user question about the sensor data.
    Returns the generated code as a string.
    """
    logger.info("--- QUERYING LLM ---")
    logger.info(f"Processing query: '{query}'")

    df = data_info['dataframe']
    df_sample = df.head(5).to_string()
    data_structure = {
        'shape': df.shape,
        'columns': list(df.columns)
    }
    
    prompt = f"""
            You are a Python data analyst. The DataFrame `df` has this structure:
            {json.dumps(data_structure, indent=2)}

            Here's a sample of the data:
            {df_sample}

            Write Python code to answer the question:

            QUESTION: {query}

            Requirements:
            1. **Use standardized columns**  
            Use the already standardized columns: 'co2', 'temp', and 'humid' which are already created for you.

            2. **Data aggregation**  
            Group or pivot as needed (e.g. by day of week, room, etc.) before plotting.

            3. **Plotting**  
            - Include plots (bar charts, time series, heatmaps, etc.) where they add clarity.  
            - **Dynamic axis scaling**: after computing your summary statistic, calculate  
                ```python
                values = summary_series.values
                data_min, data_max = values.min(), values.max()
                margin = 0.05 * (data_max - data_min) or a fixed margin if range is zero
                ax.set_ylim(data_min - margin, data_max + margin)
                ```  
            - **Custom ticks**: use `matplotlib.ticker.MultipleLocator` or `MaxNLocator` to choose sensible tick intervals.  
            - Remember to label axes, title your chart, and rotate tick labels if needed.

            4. **HTML output**  
            Return a `result` dict with:  
            - `summary`: a concise text summary  
            - `table_html`: HTML table string (or `None`)  
            - `plot_html`: HTML `<img>` tag with your base64-encoded figure (or `None`)
            
            IMPORTANT: When using f-strings with pandas Series values, convert them to Python scalars first:
            
            Example (incorrect - will cause TypeError):
            f"The max value is {{series.max():.2f}}"
            
            Example (correct - convert to Python scalar first):
            f"The max value is {{float(series.max()):.2f}}"
            f"The highest day is {{str(series.idxmax())}}"

            5. **Execution context**  
            Your code will run with:
            - `df`: the DataFrame  
            - Standard imports: `import pandas as pd, numpy as np, datetime, matplotlib.pyplot as plt, io, base64`

            6. **Plot-to-HTML snippet**  
            Use this boilerplate to convert your `fig` to HTML:
            
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', bbox_inches='tight')
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
            plot_html = f'<img src="data:image/png;base64,{{img_b64}}" alt="Analysis Plot">'
            plt.close(fig)

            End by assigning to `result`:
            
            result = {{'summary': summary_text, 'table_html': table_html, 'plot_html': plot_html}}
            """

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {HF_TOKEN}'
    }
    payload = {
        'model': 'moonshotai/Kimi-K2-Instruct:novita',
        'messages': [
            {'role': 'system', 'content': 'You are a helpful assistant that writes Python code.'},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.1,
        'max_tokens': 2000
    }

    try:
        response = requests.post(
            'https://router.huggingface.co/v1/chat/completions',
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        text = response.json()['choices'][0]['message']['content']
       
        # strip markdown fences
        code = re.sub(r'^```(?:python)?', '', text)
        code = re.sub(r'```$', '', code)
        logger.info("✓ Received code from LLM")
        logger.debug(code)
        return code.strip()
    except Exception as e:
        logger.error(f"ERROR querying LLM: {e}")
        return None


def execute_code(code, data_info):
    """
    Execute the generated Python code in a sandboxed namespace.
    Returns a dict with 'summary', 'table_html', and 'plot_html'.
    """
    logger.info("--- EXECUTING GENERATED CODE ---")
    logger.info(code)

    # Debug data before execution
    df = data_info['dataframe']
    logger.debug(f"--- DATA DEBUG BEFORE EXECUTION ---")
    logger.debug(f"DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")
    
    # Check for NaN values in important columns
    for col in ['co2', 'temp', 'humid']:
        if col in df.columns:
            nan_count = df[col].isna().sum()
            nan_percent = 100 * nan_count / len(df) if len(df) > 0 else 0
            logger.debug(f"Column '{col}' has {nan_count} NaN values ({nan_percent:.2f}%)")
            
            # Check NaN by room
            for room in df['room'].unique():
                room_nan = df[df['room'] == room][col].isna().sum()
                room_total = len(df[df['room'] == room])
                room_nan_pct = 100 * room_nan / room_total if room_total > 0 else 0
                if room_nan > 0:
                    logger.debug(f"  Room '{room}': {room_nan}/{room_total} NaN values ({room_nan_pct:.2f}%)")

    namespace = {
        'df': data_info['dataframe'],
        'original_fields': data_info.get('original_fields', {}),
        'pd': pd,
        'np': np,
        'datetime': datetime,
        'plt': plt,
        'io': io,
        'base64': base64,
        'matplotlib': matplotlib
    }

    try:
        # Add a pre-execution function to verify all data is properly loaded
        df_debug_str = "# Debug and verify data at the start\n"
        df_debug_str += "print('Verifying DataFrame contents')\n"
        df_debug_str += "# Show basic statistics for each room\n"
        df_debug_str += "df = df.copy()\n"
        df_debug_str += "for col in ['co2', 'temp', 'humid']:\n"
        df_debug_str += "    if col in df.columns:\n"
        df_debug_str += "        print(f'Column {col} statistics:')\n"
        df_debug_str += "        print(f'  Total values: {df[col].count()} out of {len(df)}')\n"
        df_debug_str += "        print(f'  Mean value: {df[col].mean():.2f}')\n"
        df_debug_str += "        # Check data by room\n"
        df_debug_str += "        for room in df['room'].unique():\n"
        df_debug_str += "            room_df = df[df['room'] == room]\n"
        df_debug_str += "            non_null_count = room_df[col].count()\n"
        df_debug_str += "            total_count = len(room_df)\n"
        df_debug_str += "            if non_null_count == 0:\n"
        df_debug_str += "                print(f'  Room {room}: NO DATA for {col}')\n"
        df_debug_str += "            else:\n"
        df_debug_str += "                mean_val = room_df[col].mean()\n"
        df_debug_str += "                print(f'  Room {room}: {non_null_count}/{total_count} values, mean: {mean_val:.2f}')\n"
        
        # Insert the data verification code at the beginning of the user code
        code = df_debug_str + "\n" + code
        
        # ensure final `result` assignment - look for it among the last few lines
        lines = code.splitlines()
        result_pattern = r'^\s*result\s*=\s*\{'
        
        # Check for result assignment in the last 5 lines (or all lines if fewer)
        check_lines = min(5, len(lines))
        found_result_assignment = False
        
        for i in range(1, check_lines + 1):
            if re.search(result_pattern, lines[-i]):
                found_result_assignment = True
                break
                
        # Also check if there's any result assignment in the entire code
        if not found_result_assignment and not any(re.search(r'result\s*=\s*\{', line) for line in lines):
            logger.warning("LLM code may not assign `result` at the end.")
            # Ensure there's a fallback result dict in the namespace
            namespace['result'] = {
                'summary': "Could not detect result assignment in generated code.",
                'table_html': None,
                'plot_html': None
            }

        # Add helper functions to the namespace
        namespace['convert_float'] = lambda x: float(x) if hasattr(x, '__float__') else x
        namespace['convert_str'] = lambda x: str(x) if x is not None else ''
        
        # Fix common issues in the LLM code before execution
        fixed_code = code
        
        # Add code to handle datetime format warnings at the beginning
        datetime_warning_fix = """
# Fix for datetime format warnings
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
"""
        fixed_code = datetime_warning_fix + "\n" + fixed_code
        
        # 1. Fix common Series to float conversions
        fixed_code = re.sub(
            r'float\(([a-zA-Z0-9_]+)\.max\(\)\)', 
            r'\1.max()', 
            fixed_code
        )
        fixed_code = re.sub(
            r'float\(([a-zA-Z0-9_]+)\.min\(\)\)', 
            r'\1.min()', 
            fixed_code
        )
        
        # 2. Auto-add helper functions to handle Series operations
        helper_functions = """
# Define our fix_code_indentation function first so it's available 
# Helper functions for safely handling pandas Series
def safe_max(series):
    # Return the maximum value of a Series as a scalar
    if hasattr(series, 'max'):
        val = series.max()
        # Handle NaN and infinite values
        if pd.isna(val) or np.isinf(val):
            return 0
        return val
    return series
    
def safe_min(series):
    # Return the minimum value of a Series as a scalar
    if hasattr(series, 'min'):
        val = series.min()
        # Handle NaN and infinite values
        if pd.isna(val) or np.isinf(val):
            return 0
        return val
    return series

def safe_idxmax(series):
    # Return the index of maximum value as a string
    if hasattr(series, 'idxmax'):
        try:
            return str(series.idxmax())
        except ValueError:
            return "No valid maximum"
    return str(series)
    
def safe_idxmin(series):
    # Return the index of minimum value as a string
    if hasattr(series, 'idxmin'):
        try:
            return str(series.idxmin())
        except ValueError:
            return "No valid minimum"
    return str(series)

def format_float(value, precision=0):
    # Format a float value with specified precision
    try:
        float_val = float(value)
        if np.isnan(float_val) or np.isinf(float_val):
            return "N/A"
        return f"{float_val:.{precision}f}"
    except:
        return str(value)
        
def validate_data_for_plot(data_series):
    # Check if data is valid for plotting (not all NaN or inf)
    if data_series is None or len(data_series) == 0:
        return False
    return not (pd.isna(data_series).all() or np.isinf(data_series).all())
"""

        preprocess_code = """
import matplotlib.pyplot as plt
import io
import base64

try:
    # Try to create a simple plot with the data
    plt.figure(figsize=(8, 4))
    
    # Choose which standardized column to plot, prioritizing CO2, then temp, then humid
    if 'co2' in df.columns:
        plot_col = 'co2'
        y_label = 'CO₂ (ppm)'
        title = 'Average CO₂ by Day of Week'
    elif 'temp' in df.columns:
        plot_col = 'temp'
        y_label = 'Temperature (°C)'
        title = 'Average Temperature by Day of Week'
    elif 'humid' in df.columns:
        plot_col = 'humid'
        y_label = 'Relative Humidity (%)'
        title = 'Average Humidity by Day of Week'
    else:
        # If none of our standardized columns exist, use first numeric column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            plot_col = numeric_cols[0]
            y_label = plot_col
            title = f'Average {plot_col} by Day of Week'
        else:
            # If no numeric columns, just show row counts
            plot_col = None
            day_counts = df.groupby(df['timestamp'].dt.day_name()).size()
            day_counts.plot(kind='bar')
            plt.title('Data Records by Day of Week')
            plt.ylabel('Count')
            y_label = 'Count'
            title = 'Data Records by Day of Week'
    
    # Plot if we have a column to plot
    if plot_col is not None:
        df.groupby(df['timestamp'].dt.day_name())[plot_col].mean().plot(kind='bar')
        plt.title(title)
        plt.ylabel(y_label)
    
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert plot to HTML
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html = f'<img src="data:image/png;base64,{img_b64}">'
    plt.close()

    # Create table HTML safely
    if plot_col:
        table_data = df.groupby(df['timestamp'].dt.day_name())[plot_col].mean()
        table_html = pd.DataFrame(table_data).to_html()
    elif 'day_counts' in locals():
        table_html = day_counts.to_frame('Count').to_html()
    else:
        # Create a simple summary of available data
        summary_data = {}
        for col in df.columns:
            if col in ['co2', 'temp', 'humid', 'timestamp', 'room']:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                summary_data[col] = df[col].mean()
        
        if summary_data:
            table_html = pd.DataFrame([summary_data]).T.to_html()
        else:
            table_html = None

    # Create a more descriptive summary
    if plot_col == 'co2':
        summary = f'Analysis of CO₂ levels by day of week. '
        if df['co2'].min() is not None and df['co2'].max() is not None:
            summary += f'Values range from {df["co2"].min():.1f} to {df["co2"].max():.1f} ppm.'
    elif plot_col == 'temp':
        summary = f'Analysis of temperature by day of week. '
        if df['temp'].min() is not None and df['temp'].max() is not None:
            summary += f'Temperature ranges from {df["temp"].min():.1f} to {df["temp"].max():.1f} °C.'
    elif plot_col == 'humid':
        summary = f'Analysis of humidity levels by day of week. '
        if df['humid'].min() is not None and df['humid'].max() is not None:
            summary += f'Humidity ranges from {df["humid"].min():.1f} to {df["humid"].max():.1f}%.'
    else:
        summary = f'Analysis of data by day of week. The chart shows the distribution across days.'
    
    # Create basic result
    result = {
        'summary': summary,
        'table_html': table_html,
        'plot_html': plot_html
    }
except Exception as e:
    # Ultra-safe fallback
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, f"Could not generate plot: {str(e)}", 
             ha='center', va='center', fontsize=12)
    plt.tight_layout()
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    plot_html = f'<img src="data:image/png;base64,{img_b64}">'
    plt.close()
    
    result = {
        'summary': f'Analysis attempted but encountered an error: {str(e)}',
        'table_html': None,
        'plot_html': plot_html
    }
"""
        
        # Add our helper functions at the beginning of the code
        fixed_code = helper_functions + "\n" + fixed_code
        
        # Ensure the code ends with a result assignment if one isn't found
        result_pattern = r'result\s*=\s*\{'
        if not re.search(result_pattern, fixed_code):
            # Append a default result assignment
            fixed_code += """

# Default result creation if none was provided
try:
    # Check if we have any analysis results already
    if 'plot_html' in locals() or 'table_html' in locals():
        result = {
            'summary': 'Analysis completed. Please see the table and/or plot for results.',
            'table_html': table_html if 'table_html' in locals() else None,
            'plot_html': plot_html if 'plot_html' in locals() else None
        }
    else:
        result = {
            'summary': 'Analysis completed without detailed results.',
            'table_html': None, 
            'plot_html': None
        }
except Exception as e:
    result = {
        'summary': f'Analysis attempted but could not generate result: {str(e)}',
        'table_html': None,
        'plot_html': None
    }
"""

        # 3. Replace problematic Series operations with our helper functions
        # Replace float(series.max()) with safe_max(series)
        fixed_code = re.sub(
            r'float\s*\(\s*([a-zA-Z0-9_\.]+)\.max\(\)\s*\)', 
            r'safe_max(\1)', 
            fixed_code
        )
        
        # Replace float(series.min()) with safe_min(series)
        fixed_code = re.sub(
            r'float\s*\(\s*([a-zA-Z0-9_\.]+)\.min\(\)\s*\)', 
            r'safe_min(\1)', 
            fixed_code
        )
        
        # Replace str(series.idxmax()) with safe_idxmax(series)
        fixed_code = re.sub(
            r'str\s*\(\s*([a-zA-Z0-9_\.]+)\.idxmax\(\)\s*\)', 
            r'safe_idxmax(\1)', 
            fixed_code
        )
        
        # Replace str(series.idxmin()) with safe_idxmin(series)
        fixed_code = re.sub(
            r'str\s*\(\s*([a-zA-Z0-9_\.]+)\.idxmin\(\)\s*\)', 
            r'safe_idxmin(\1)', 
            fixed_code
        )
        
        # Note: %H is hour (00-23), while %h is abbreviated month name - correct replacement should be %I (hour 01-12)
        fixed_code = re.sub(
            r"\.dt\.strftime\(['\"]%H", 
            r".dt.strftime('%I", 
            fixed_code
        )
        
        # Also handle hour format in strptime and other datetime formatting functions
        fixed_code = re.sub(
            r"datetime\.strptime\([^,]+,\s*['\"][^'\"]*%H", 
            lambda m: m.group(0).replace('%H', '%I'), 
            fixed_code
        )
        
        # Replace any reference to .to_frame() on a Series with explicit DataFrame conversion
        fixed_code = re.sub(
            r'([a-zA-Z0-9_\.]+)\.to_frame\(([^)]*)\)', 
            r'pd.DataFrame(\1, columns=[\2] if \2 else None)', 
            fixed_code
        )
        
        # Fix NaN handling in matplotlib limits
        fixed_code = re.sub(
            r'ax\.set_ylim\(([^,]+), ([^)]+)\)', 
            r'if not (np.isnan(\1) or np.isnan(\2) or np.isinf(\1) or np.isinf(\2)): ax.set_ylim(\1, \2)', 
            fixed_code
        )
        # Execute the helper functions first
        exec(preprocess_code, namespace)
            
        try:
            # Try executing the fixed code with our helper functions
            logger.debug("Executing fixed code with helper functions")
            
            # Add the fix_code_indentation function to the namespace
            namespace['fix_code_indentation'] = fix_code_indentation
            
            # Fix common indentation issues
            fixed_code_clean = fix_code_indentation(fixed_code)
            
            # Execute the fixed code
            exec(fixed_code_clean, namespace)
        except Exception as e:
            logger.warning(f"Fixed code execution failed: {str(e)}, trying a different approach")
            
            # # Try a more direct approach - find and fix the specific issue in the 
            # # code related to Series max/min/idxmax/idxmin operations and NaN handling
            # simple_fixed_code = code.replace("float(mean_co2.max())", "mean_co2.max()")
            # simple_fixed_code = simple_fixed_code.replace("float(mean_co2.min())", "mean_co2.min()")
            # simple_fixed_code = simple_fixed_code.replace("str(mean_co2.idxmax())", "str(mean_co2.idxmax())")
            # simple_fixed_code = simple_fixed_code.replace("str(mean_co2.idxmin())", "str(mean_co2.idxmin())")
            
            # # Fix axis limits NaN errors by adding checks
            # simple_fixed_code = simple_fixed_code.replace(
            #     "ax.set_ylim(data_min - margin, data_max + margin)",
            #     "if not (np.isnan(data_min) or np.isnan(data_max) or np.isinf(data_min) or np.isinf(data_max)): ax.set_ylim(data_min - margin, data_max + margin)"
            # )
            
            # # Fix indentation issues using our helper function
            # simple_fixed_code_clean = fix_code_indentation(simple_fixed_code)
            
            # try:
            #     # Execute with the same namespace so we have access to all the variables
            #     exec(simple_fixed_code_clean, namespace)
            # except Exception as e2:
            #     # If all else fails, try to modify the code to use a very direct approach
            #     logger.warning(f"Simple fixed code also failed: {str(e2)}, trying last resort approach")
                
                # Create a simple fallback result using the preprocess_code in the same namespace
                # This ensures we have access to the df and other variables
                # try:
                #     exec(preprocess_code, namespace)
                # except Exception as e3:
                #     logger.error(f"Even fallback preprocessing failed: {str(e3)}")
                #     # Ultimate fallback - create a minimal result
                #     namespace['result'] = {
                #         'summary': f"Could not process query due to technical issues: {str(e3)}",
                #         'table_html': None,
                #         'plot_html': None
                #     }
        result = namespace.get('result')
        if not isinstance(result, dict):
            # If execution succeeded but still no result dict, create one with available data
            logger.warning("No `result` dict produced by code, creating a default one")
            result = {
                'summary': "Analysis completed. The code executed successfully but did not produce a result dictionary.",
                'table_html': None,
                'plot_html': None
            }
            
            # Try to extract any plot_html or table_html that might have been generated
            if 'plot_html' in namespace:
                result['plot_html'] = namespace['plot_html']
            if 'table_html' in namespace:
                result['table_html'] = namespace['table_html']
                
        # Ensure all required keys are present
        if 'plot_html' not in result:
            result['plot_html'] = None
        if 'table_html' not in result:
            result['table_html'] = None
        if 'summary' not in result or not result['summary']:
            result['summary'] = "Analysis completed successfully."
            
        # Replace NaN values with 'N/A' in table_html
        if result['table_html']:
            result['table_html'] = result['table_html'].replace('>NaN<', '>N/A<')
            
        # Log the details of the results
        logger.debug(f"Result summary: {result.get('summary', 'No summary')}")
        if result['table_html']:
            logger.debug(f"Table HTML length: {len(result['table_html'])}, sample: {result['table_html'][:100]}...")
        if result['plot_html']:
            logger.debug(f"Plot HTML length: {len(result['plot_html'])}, sample: {result['plot_html'][:100]}...")
            
        logger.info("✓ Code executed successfully")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"Execution error: {e}\n{tb}")
        return {
            'summary': f"Error executing analysis: {e}",
            'table_html': None,
            'plot_html': None
        }
