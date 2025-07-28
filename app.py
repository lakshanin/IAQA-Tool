from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
import os
from agent import load_sensor_data, get_code_from_llm, execute_code

# load HF_TOKEN from .env
load_dotenv()
HF_TOKEN = os.getenv('HF')


# static files
app = Flask(__name__, static_folder='static')

@app.route('/')
def index():
    # serve the React entry-point
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json.get('query','').strip()
    if not user_query:
        return jsonify({'success': False, 'summary': 'Empty query', 'table_html': None, 'plot_html': None}), 400

    data_info = load_sensor_data()
    if not data_info:
        return jsonify({'success': False, 'summary': 'No sensor data found.', 'table_html': None, 'plot_html': None})

    code = get_code_from_llm(user_query, data_info, HF_TOKEN)
    if not code:
        return jsonify({'success': False, 'summary': 'LLM did not return code.', 'table_html': None, 'plot_html': None})

    result = execute_code(code, data_info)
    return jsonify({
        'success':    True,
        'summary':    result.get('summary',''),
        'table_html': result.get('table_html', None),
        'plot_html':  result.get('plot_html', None)
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
