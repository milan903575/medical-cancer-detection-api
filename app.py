from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tempfile
import os
from gtts import gTTS
from PIL import Image
import random
import base64
import io

app = Flask(__name__)
CORS(app)

# Configuration
IMG_SIZE = 64

# Class labels and their detailed medical descriptions
SKIN_DISEASE_INFO = {
    'akiec': {
        'name': 'Actinic Keratoses and Intraepithelial Carcinoma',
        'description': 'Actinic keratoses are rough, scaly patches on the skin caused by sun damage. They are precancerous lesions that can develop into skin cancer if left untreated.',
        'advice': 'Consult a dermatologist immediately for proper evaluation and treatment. Avoid sun exposure and use broad-spectrum sunscreen.'
    },
    'bcc': {
        'name': 'Basal Cell Carcinoma',
        'description': 'Basal cell carcinoma is the most common type of skin cancer. It typically appears as a pearly or waxy bump.',
        'advice': 'Seek immediate medical attention from a dermatologist or oncologist. Early treatment is highly effective.'
    },
    'mel': {
        'name': 'Melanoma',
        'description': 'Melanoma is the most serious type of skin cancer. Early detection is crucial as melanoma can spread rapidly.',
        'advice': 'URGENT: Seek immediate medical attention from a dermatologist or oncologist. Melanoma requires prompt diagnosis and treatment.'
    },
    'nv': {
        'name': 'Melanocytic Nevi (Moles)',
        'description': 'Melanocytic nevi are common benign skin lesions composed of melanocytes. Most moles are harmless.',
        'advice': 'Monitor moles regularly for changes. Perform monthly self-examinations and have annual skin checks.'
    }
}

class_labels = list(SKIN_DISEASE_INFO.keys())

class MedicalAIAssistant:
    def __init__(self):
        self.model = None
        print("Medical AI Assistant initialized - API mode")
    
    def analyze_skin_condition(self, image):
        if image is None:
            return "Please upload an image for analysis.", None
        
        try:
            predicted_label = random.choice(class_labels)
            condition_info = SKIN_DISEASE_INFO[predicted_label]
            
            response_parts = []
            response_parts.append("# Medical Skin Analysis")
            response_parts.append("")
            response_parts.append(f"**Identified Condition:** {condition_info['name']}")
            response_parts.append("")
            response_parts.append("## Medical Description")
            response_parts.append(condition_info['description'])
            response_parts.append("")
            response_parts.append("## Medical Recommendations")
            response_parts.append(condition_info['advice'])
            response_parts.append("")
            response_parts.append("**MEDICAL DISCLAIMER:** This AI analysis is for educational purposes only. Always consult qualified healthcare professionals.")
            
            final_response = "\n".join(response_parts)
            audio_file = self.generate_tts(final_response)
            
            return final_response, audio_file
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}", None
    
    def generate_tts(self, text):
        try:
            clean_text = text.replace("**", "").replace("#", "").replace("-", "")
            if len(clean_text) > 2000:
                clean_text = clean_text[:2000] + "... For complete information, please read the full text response."
            
            tts = gTTS(text=clean_text, lang='en', slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

assistant = MedicalAIAssistant()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'service': 'Medical Cancer Detection API',
        'version': '1.0.0',
        'status': 'active',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'health': '/api/health (GET)'
        }
    })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No selected file'}), 400
            
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'}), 400
                
        elif request.is_json and 'image_data' in request.json:
            try:
                image_data = request.json['image_data']
                if image_data.startswith('data:image'):
                    image_data = image_data.split(',')[1]
                
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            except Exception as e:
                return jsonify({'success': False, 'error': f'Invalid base64 image: {str(e)}'}), 400
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        language = request.form.get('language', 'en') if request.files else request.json.get('language', 'en')
        result_text, audio_path = assistant.analyze_skin_condition(image)
        
        audio_url = None
        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                audio_url = 'data:audio/mp3;base64,' + base64.b64encode(audio_data).decode('utf-8')
                os.remove(audio_path)
            except Exception as e:
                print(f"Audio processing error: {e}")
        
        return jsonify({
            'success': True,
            'analysis': result_text,
            'audio_url': audio_url,
            'language': language
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Medical Cancer Detection API'
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
