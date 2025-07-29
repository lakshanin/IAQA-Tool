# Core imports needed for utility functions
import pandas as pd
import numpy as np

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

# Helper functions for safely handling pandas Series
def safe_max(series):
    """Return the maximum value of a Series as a scalar"""
    if hasattr(series, 'max'):
        val = series.max()
        # Handle NaN and infinite values
        if pd.isna(val) or np.isinf(val):
            return 0
        return val
    return series
    
def safe_min(series):
    """Return the minimum value of a Series as a scalar"""
    if hasattr(series, 'min'):
        val = series.min()
        # Handle NaN and infinite values
        if pd.isna(val) or np.isinf(val):
            return 0
        return val
    return series

def safe_idxmax(series):
    """Return the index of maximum value as a string"""
    if hasattr(series, 'idxmax'):
        try:
            return str(series.idxmax())
        except ValueError:
            return "No valid maximum"
    return str(series)
    
def safe_idxmin(series):
    """Return the index of minimum value as a string"""
    if hasattr(series, 'idxmin'):
        try:
            return str(series.idxmin())
        except ValueError:
            return "No valid minimum"
    return str(series)

def format_float(value, precision=0):
    """Format a float value with specified precision"""
    try:
        float_val = float(value)
        if np.isnan(float_val) or np.isinf(float_val):
            return "N/A"
        return f"{float_val:.{precision}f}"
    except:
        return str(value)
        
def validate_data_for_plot(data_series):
    """Check if data is valid for plotting (not all NaN or inf)"""
    if data_series is None or len(data_series) == 0:
        return False
    return not (pd.isna(data_series).all() or np.isinf(data_series).all())

# Simple data conversion utilities
def convert_float(x):
    """Convert a value to float if possible, otherwise return the original value"""
    return float(x) if hasattr(x, '__float__') else x

def convert_str(x):
    """Convert a value to string if not None, otherwise return empty string"""
    return str(x) if x is not None else ''
