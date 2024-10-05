from flask import Flask, request, jsonify, send_from_directory
import whisper
import os
import re
import subprocess
import threading

app = Flask(__name__, static_folder='static')

# Cargar el modelo Whisper
model = whisper.load_model("base")

# Almacenar las transcripciones en un diccionario temporal
transcription_results = {}

# Ruta para servir el archivo HTML
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Función para normalizar el nombre del archivo
def normalize_filename(filename):
    filename = re.sub(r'[^\w\s-]', '', filename)
    return re.sub(r'\s+', '_', filename)

# Función que realiza la transcripción de audio
def transcribir_audio_async(app, audio_file, audio_path, wav_path, transcription_id):
    with app.app_context():
        try:
            print(f"Convirtiendo {audio_file.filename} a formato WAV...")
            subprocess.run(['ffmpeg', '-i', audio_path, wav_path], check=True)
            print(f"Archivo convertido a .wav: {wav_path}")
            
            print("Iniciando transcripción con Whisper...")
            result = model.transcribe(wav_path)
            transcription_results[transcription_id] = result['text']  # Almacenar transcripción
            
            # Borrar archivos temporales
            os.remove(wav_path)
            os.remove(audio_path)
            
            print(f"Transcripción completada: {result['text']}")
        except Exception as e:
            transcription_results[transcription_id] = f"Error en la transcripción: {str(e)}"
            print(f"Error en el proceso de transcripción: {str(e)}")

# Ruta para manejar la transcripción de audio
@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        print("Error: No se envió ningún archivo en la solicitud.")
        return jsonify({"error": "No file provided"}), 400
    
    audio_file = request.files['file']
    
    print("Archivo recibido:", audio_file.filename)
    normalized_filename = normalize_filename(audio_file.filename)
    print("Nombre del archivo normalizado:", normalized_filename)

    if not audio_file.filename.endswith(('.wav', '.mp3', '.m4a')):
        print("Error: Formato de archivo no soportado.")
        return jsonify({"error": "Unsupported file format"}), 400

    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    audio_path = os.path.join("temp", normalized_filename)
    audio_file.save(audio_path)
    
    wav_path = os.path.join("temp", os.path.splitext(normalized_filename)[0] + ".wav")

    # Crear un identificador único para la transcripción
    transcription_id = re.sub(r'\W+', '', normalized_filename)

    # Iniciar el proceso de transcripción en un hilo separado
    thread = threading.Thread(target=transcribir_audio_async, args=(app, audio_file, audio_path, wav_path, transcription_id))
    thread.start()

    # Devolver el ID de la transcripción para que el frontend pueda hacer polling
    return jsonify({"status": "Transcripción en progreso...", "transcription_id": transcription_id}), 202

# Nueva ruta para consultar el estado de la transcripción
@app.route('/transcription_status/<transcription_id>', methods=['GET'])
def get_transcription_status(transcription_id):
    # Comprobar si la transcripción está completa
    if transcription_id in transcription_results:
        transcription = transcription_results.pop(transcription_id)  # Remover y devolver el resultado
        return jsonify({"transcription": transcription}), 200
    else:
        return jsonify({"status": "En progreso..."}), 202

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
