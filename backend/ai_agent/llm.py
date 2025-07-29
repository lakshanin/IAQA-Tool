# Core imports
import json
import re
import matplotlib
import requests
matplotlib.use('Agg')  
from backend.logs import logger 


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
