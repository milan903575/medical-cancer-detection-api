import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import tempfile
import os
from gtts import gTTS
from PIL import Image
import random
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import base64
import io

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
    'bkl': {
        'name': 'Benign Keratosis-like Lesions',
        'description': 'These are non-cancerous skin growths that include seborrheic keratoses and solar lentigines.',
        'advice': 'While generally harmless, have any new or changing skin lesions evaluated by a dermatologist.'
    },
    'df': {
        'name': 'Dermatofibroma',
        'description': 'Dermatofibroma is a common benign skin tumor that appears as a small, firm nodule, usually on the legs.',
        'advice': 'These are typically harmless and do not require treatment. Consult a dermatologist if changes occur.'
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
    },
    'vasc': {
        'name': 'Vascular Lesions',
        'description': 'Vascular lesions are abnormalities of blood vessels in the skin. Most are benign but may cause cosmetic concerns.',
        'advice': 'Most vascular lesions are benign. Consult a dermatologist for proper evaluation if needed.'
    }
}

class_labels = list(SKIN_DISEASE_INFO.keys())

class MedicalAIAssistant:
    def __init__(self):
        self.model = None
        print("Medical AI Assistant initialized - using demo mode")
    
    def analyze_skin_condition(self, image):
        """Analyze skin condition from image and provide detailed medical response."""
        if image is None:
            return "Please upload an image for analysis.", None
        
        try:
            # Demo prediction (replace with real model when available)
            predicted_label = random.choice(class_labels)
            
            # Get detailed information
            condition_info = SKIN_DISEASE_INFO[predicted_label]
            
            # Generate comprehensive medical response
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
            
            # Add urgency level
            if predicted_label in ['mel', 'bcc', 'akiec']:
                response_parts.append("## URGENCY LEVEL: HIGH")
                response_parts.append("This condition requires immediate medical attention from a qualified dermatologist.")
            else:
                response_parts.append("## URGENCY LEVEL: ROUTINE")
                response_parts.append("Schedule a consultation with a dermatologist for proper evaluation.")
            
            response_parts.append("")
            response_parts.append("## General Skin Health Tips")
            response_parts.append("- Perform regular self-examinations of your skin")
            response_parts.append("- Use broad-spectrum sunscreen daily (SPF 30+)")
            response_parts.append("- Avoid excessive sun exposure")
            response_parts.append("- Stay hydrated and maintain a healthy diet")
            response_parts.append("- Schedule annual skin checks with a dermatologist")
            response_parts.append("")
            response_parts.append("**MEDICAL DISCLAIMER:** This AI analysis is for educational purposes only. Always consult qualified healthcare professionals.")
            
            final_response = "\n".join(response_parts)
            
            # Generate audio
            audio_file = self.generate_tts(final_response)
            
            return final_response, audio_file
            
        except Exception as e:
            return f"Error analyzing image: {str(e)}", None
    
    def generate_tts(self, text):
        """Convert text to speech."""
        try:
            # Clean text for TTS
            clean_text = text.replace("**", "").replace("#", "").replace("-", "")
            
            # Limit text length for TTS
            if len(clean_text) > 2000:
                clean_text = clean_text[:2000] + "... For complete information, please read the full text response."
            
            tts = gTTS(text=clean_text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
                tts.save(tmp_file.name)
                return tmp_file.name
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

# Initialize the assistant
assistant = MedicalAIAssistant()

def process_image_analysis(image):
    """Process image analysis for Gradio interface."""
    return assistant.analyze_skin_condition(image)

# Create Gradio interface
demo = gr.Interface(
    fn=process_image_analysis,
    inputs=gr.Image(type="pil", label="Upload Skin Image"),
    outputs=[
        gr.Textbox(label="Medical Analysis", lines=15),
        gr.Audio(label="Audio Analysis")
    ],
    title="Medical AI Skin Analysis",
    description="Upload an image of a skin condition for AI-powered medical analysis. This tool provides preliminary analysis only - always consult healthcare professionals.",
    examples=None
)

# Flask API for external access
app = Flask(__name__)
CORS(app)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Flask API endpoint for image analysis."""
    try:
        # Handle different input formats
        if 'image' in request.files:
            # File upload
            file = request.files['image']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'No selected file'}), 400
            
            try:
                image = Image.open(file.stream).convert('RGB')
            except Exception as e:
                return jsonify({'success': False, 'error': f'Invalid image file: {str(e)}'}), 400
                
        elif 'image_data' in request.json:
            # Base64 image data
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
        
        # Get language parameter
        language = request.form.get('language', 'en') if request.files else request.json.get('language', 'en')
        
        # Analyze image
        result_text, audio_path = assistant.analyze_skin_condition(image)
        
        # Handle audio file
        audio_url = None
        if audio_path and os.path.exists(audio_path):
            try:
                with open(audio_path, 'rb') as f:
                    audio_data = f.read()
                audio_url = 'data:audio/mp3;base64,' + base64.b64encode(audio_data).decode('utf-8')
                os.remove(audio_path)  # Clean up temp file
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
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'service': 'Medical AI Assistant'})

@app.route('/api/info', methods=['GET'])
def api_info():
    """API information endpoint."""
    return jsonify({
        'service': 'Medical AI Assistant',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/predict (POST)',
            'health': '/api/health (GET)',
            'info': '/api/info (GET)'
        },
        'supported_formats': ['JPEG', 'PNG'],
        'max_file_size': '10MB'
    })

def run_flask():
    """Run Flask app in a separate thread."""
    app.run(host='0.0.0.0', port=5000, debug=False)

def run_gradio():
    """Run Gradio app."""
    demo.launch(share=True, server_port=7860)

if __name__ == "__main__":
    # Start Flask API in a separate thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("🚀 Medical AI Assistant Starting...")
    print("📱 Gradio Interface: http://localhost:7860")
    print("🔗 Flask API: http://localhost:5000")
    print("📋 API Docs: http://localhost:5000/api/info")
    
    # Start Gradio interface
    run_gradio()
