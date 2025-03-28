import os
import io
from flask import Flask, render_template, request, jsonify, send_file
import google.generativeai as genai
import speech_recognition as sr
from pydub import AudioSegment
from gtts import gTTS

app = Flask(__name__)

# Configure Google AI API
GOOGLE_API_KEY = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
genai.configure(api_key=GOOGLE_API_KEY)
MODEL_NAME = 'gemini-1.5-flash'
model = genai.GenerativeModel(MODEL_NAME)

PERSONA_CONTEXT = """
You are Anjan Mondal, a 25-year-old data scientist at o9 Solutions. Your defining trait is perseverance, allowing you to tackle complex problems with persistence and curiosity. Your growth areas include machine learning, statistics, and low-level programming, reflecting your drive for deep technical expertise.

You are analytical yet practical, constantly updating your knowledge by reading books, exploring articles, and experimenting hands-on. While some coworkers might misunderstand aspects of your approach, you bring a thoughtful, structured mindset to problem-solving.

Instructions for AI Response Generation:

Provide technical insights with clarity and depth.

Relate answers to practical applications in data science, statistics, and programming.

Maintain a growth-oriented perspective, encouraging continuous learning.

Engage in thoughtful, nuanced discussion rather than surface-level responses.

Keep answers short to one or two sentences, unless asked to elaborate.

"""

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file found'}), 400

    audio_file = request.files['audio']
    
    # Convert audio file to WAV format for SpeechRecognition
    audio = AudioSegment.from_file(audio_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)

    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_io) as source:
        audio_data = recognizer.record(source)
        try:
            transcript = recognizer.recognize_google(audio_data)
            return jsonify({'transcript': transcript})
        except sr.UnknownValueError:
            return jsonify({'transcript': '', 'error': 'Could not understand audio'})
        except sr.RequestError as e:
            return jsonify({'transcript': '', 'error': f'Speech recognition failed: {e}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    
    full_prompt = f"{PERSONA_CONTEXT}\n\nUser Question: {user_message}\n\nResponse:"
    
    try:
        response = model.generate_content(full_prompt)
        bot_reply = response.text
        return jsonify({'reply': bot_reply})
    except Exception as e:
        print(f"Generation error: {e}")
        return jsonify({'reply': "I'm experiencing technical difficulties.", 'error': str(e)}), 500

@app.route('/get_audio', methods=['GET'])
def get_audio():
    user_message = request.args.get('message', '')

    try:
        tts = gTTS(text=user_message, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        
        return send_file(mp3_fp, mimetype='audio/mp3', as_attachment=False)
    except Exception as e:
        print(f"Audio generation error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False)
