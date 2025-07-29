# Core imports
import os
import json
import glob
import re
import pandas as pd
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  
from backend.logs import logger 

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
