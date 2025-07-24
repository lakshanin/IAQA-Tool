from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import json
import openai
import requests
from dotenv import load_dotenv
import traceback

# Load HuggingFace token from .env
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

app = Flask(__name__, static_folder='static', template_folder='templates')

DATA_DIR = os.path.join(os.path.dirname(__file__), 'sensor-data')

LLM_MODEL = "moonshotai/Kimi-K2-Instruct:novita"
LLM_ENDPOINT = "https://router.huggingface.co/v1/chat/completions"

# Helper: Read all sensor files
def read_sensor_files():
    data = {}
    for fname in os.listdir(DATA_DIR):
        if fname.endswith('.ndjson') or fname.endswith('.txt'):
            room = fname.split('.')[0]
            with open(os.path.join(DATA_DIR, fname), 'r', encoding='utf-8') as f:
                lines = [json.loads(line) for line in f if line.strip()]
                data[room] = lines
    return data

# Helper: Call HuggingFace LLM
def call_llm(user_query, sensor_data):
    prompt = f"""
You are an expert data analyst AI agent. The user will ask questions about air quality sensor data from multiple rooms. Each room's data is a list of JSON objects, but field names for CO2, temperature, humidity, and timestamp may vary (e.g., 'co2', 'CO2 (PPM)', 'carbon_dioxide', etc). Your job is to:
- Intelligently infer and normalize these fields (do NOT hardcode mappings).
- Write Python code to answer the user's query.
- Return a summary and, if relevant, a table (as a list of dicts).

User query: {user_query}

sensor_data = {json.dumps(sensor_data)[:2000]}  # Truncated for prompt size

Respond ONLY with a Python dict with keys 'summary' (str) and optional 'table' (list of dicts). Your code will be executed in a sandbox.
"""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful Python data analyst."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.2
    }
    resp = requests.post(LLM_ENDPOINT, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    result = resp.json()
    # Try to extract code from LLM response
    code = None
    for choice in result.get('choices', []):
        msg = choice.get('message', {}).get('content', '')
        if '```python' in msg:
            code = msg.split('```python')[1].split('```')[0].strip()
        elif '```' in msg:
            code = msg.split('```')[1].split('```')[0].strip()
        else:
            code = msg.strip()
        if code:
            break
    return code

# Helper: Safe code execution

# Allow safe imports for common analysis modules
SAFE_BUILTINS = {'min', 'max', 'sum', 'len', 'range', 'sorted', 'map', 'filter', 'any', 'all', 'abs', 'round', 'enumerate', 'set', 'list', 'dict', 'str', 'int', 'float', 'bool', 'Exception', 'ImportError', 'KeyError', 'ValueError', 'TypeError', 'AttributeError'}

# Instead of a whitelist, we'll use a blocklist for potentially dangerous modules
BLOCKED_MODULES = {
    'os', 'sys', 'subprocess', 'shutil', 'socket', 'pickle', 'requests',
    'builtins', 'importlib', 'glob', 'pty', 'platform', 'ctypes',
}

import io
import sys

def safe_exec(code, sensor_data):
    local_vars = {'sensor_data': sensor_data}
    safe_globals = {k: __builtins__.__dict__[k] for k in SAFE_BUILTINS if k in __builtins__.__dict__}
    # Add Exception and ImportError explicitly if not present
    safe_globals['Exception'] = Exception
    safe_globals['ImportError'] = ImportError

    # Custom __import__ that blocks dangerous modules but allows common data science libraries
    def limited_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in BLOCKED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed for security reasons.")
        try:
            return __import__(name, globals, locals, fromlist, level)
        except ImportError:
            raise ImportError(f"Could not import module '{name}'. Make sure it's installed.")

    safe_globals['__builtins__'] = dict(safe_globals)
    safe_globals['__builtins__']['__import__'] = limited_import
    
    # Pre-import common data science libraries
    try:
        safe_globals['pandas'] = __import__('pandas')
    except ImportError:
        pass  # pandas not installed
        
    try:
        safe_globals['numpy'] = __import__('numpy')
    except ImportError:
        pass  # numpy not installed

    # Capture print output
    stdout = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = stdout
    try:
        exec(code, safe_globals, local_vars)
        result = local_vars.get('result', None)
        output = stdout.getvalue().strip()
        if not result:
            if output:
                result = {'summary': f'(No result variable set by code.)\nOutput:\n{output}'}
            else:
                result = {'summary': 'No result returned and no output.'}
        return result
    except Exception as e:
        return {'summary': f'Error in code execution: {e}\n{traceback.format_exc()}'}
    finally:
        sys.stdout = old_stdout

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query', '')
    sensor_data = read_sensor_files()
    code = call_llm(user_query, sensor_data)
    if not code:
        return jsonify({'summary': 'No code generated by LLM.'})
    result = safe_exec(code, sensor_data)
    return jsonify(result)

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
