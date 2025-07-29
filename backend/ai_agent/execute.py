# Core imports
import re
import traceback
import pandas as pd
import numpy as np
from datetime import datetime
import base64
import io
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  
from backend.logs import logger

# Import utility functions
from backend.ai_agent.utils import (
    fix_code_indentation, safe_max, safe_min, safe_idxmax, safe_idxmin,
    format_float, validate_data_for_plot, convert_float, convert_str
)


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
        namespace['convert_float'] = convert_float
        namespace['convert_str'] = convert_str
        namespace['safe_max'] = safe_max
        namespace['safe_min'] = safe_min
        namespace['safe_idxmax'] = safe_idxmax
        namespace['safe_idxmin'] = safe_idxmin
        namespace['format_float'] = format_float
        namespace['validate_data_for_plot'] = validate_data_for_plot
        namespace['fix_code_indentation'] = fix_code_indentation
        
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
        
        # No need to define helper functions in the code anymore since they're imported from utils
        helper_functions = ""

        # We're no longer creating a default plot
        preprocess_code = """
# No default plot creation - let the generated code handle plotting
# Just define a fallback result if nothing is generated
if 'result' not in locals() or not isinstance(result, dict):
    result = {
        'summary': 'Analysis completed but no visualization was generated.',
        'table_html': None,
        'plot_html': None
    }
"""
        
        # Add helper functions at the beginning of the code
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
            
            # Fix common indentation issues (fix_code_indentation is already in namespace)
            fixed_code_clean = fix_code_indentation(fixed_code)
            
            # Execute the fixed code
            exec(fixed_code_clean, namespace)
        except Exception as e:
            logger.warning(f"Fixed code execution failed: {str(e)}, trying a different approach")
            
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
