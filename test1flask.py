from flask import Flask, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com')

def ask_deepseek(message, model = "deepseek-chat"):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "content-type": "application/json"
    }
    data = {
        "model":model,
        "messages":[
            {
                "role":"user",
                "content": message
            }
        ],
        "stream": False
    }
    try:
        response = requests.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        result = response.json()
        return result['choices'][0]['message']['content']
    
    except requests.exceptions.RequestException as e:
        return f"Error al conectar con DeepSeek: {str(e)}"
    except KeyError:
        return "Error: respuesta inesperada de la API"
    
@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error":"se requiere un campo 'pregunta'"}),400
    
    question = data['question']
    model = data.get('model','deepseek-chat')
    answer = ask_deepseek(question,model)

    return jsonify({
        "question":question,
        "answer":answer,
        "model":model
    })

@app.route('/')
def home():
    return jsonify({
        "message": "API de DeepSeek",
        "endpoints": {
            "health":"/api/health {GET}",
            "ask":"/api/ask {POST}"
        }
    })

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=500)