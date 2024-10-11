from flask import Flask, request, jsonify, send_from_directory
import whisperx
import os
import re
import threading
import subprocess
import torch

app = Flask(__name__, static_folder='static')

# Almacenar las transcripciones en un diccionario temporal
transcription_results = {}

# Cache para los modelos de alineación por idioma
alignment_model_cache = {}

# Cargar el modelo de transcripción de WhisperX una sola vez
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisperx.load_model("large", device=device, compute_type="float32")
print(f"Modelo WhisperX cargado en {device}.")

# Ruta para servir el archivo HTML
@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Función para normalizar el nombre del archivo
def normalize_filename(filename):
    base, ext = os.path.splitext(filename)
    # Eliminar caracteres no deseados del nombre base
    base = re.sub(r'[^\w\s-]', '', base)
    base = re.sub(r'\s+', '_', base)
    # Retornar el nombre normalizado con la extensión original
    return base + ext

# Función para preprocesar el audio: eliminar ruido y normalizar volumen
def preprocess_audio(input_path, output_path):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-af', 'afftdn,loudnorm',  # Eliminación de ruido y normalización de volumen
            '-y', output_path
        ], check=True)
        print(f"Audio preprocesado guardado en: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error en el preprocesamiento del audio: {e}")
        raise

# Función para obtener o cargar el modelo de alineación para un idioma específico
def get_alignment_model(language_code):
    if language_code in alignment_model_cache:
        return alignment_model_cache[language_code]
    else:
        try:
            alignment_model, metadata = whisperx.load_align_model(
                language_code=language_code, device=device
            )
            alignment_model_cache[language_code] = (alignment_model, metadata)
            print(f"Modelo de alineación cargado para el idioma: {language_code}")
            return alignment_model, metadata
        except Exception as e:
            print(f"Error al cargar el modelo de alineación para {language_code}: {e}")
            raise

# Función que realiza la transcripción de audio con WhisperX
def transcribir_audio_async(audio_path, transcription_id):
    try:
        print("Iniciando preprocesamiento del audio...")
        # Definir el nombre del archivo preprocesado con extensión .wav
        preprocessed_audio_path = os.path.join("temp", f"preprocessed_{os.path.splitext(os.path.basename(audio_path))[0]}.wav")
        preprocess_audio(audio_path, preprocessed_audio_path)

        print("Iniciando transcripción con WhisperX...")
        # Transcribir el audio preprocesado
        result = model.transcribe(preprocessed_audio_path)

        # Obtener el código del idioma detectado
        language_code = result["language"]

        print(f"Idioma detectado: {language_code}")

        # Obtener el modelo de alineación para el idioma
        alignment_model, metadata = get_alignment_model(language_code)

        # Realizar la alineación
        result_aligned = whisperx.align(
            result["segments"], alignment_model, metadata, preprocessed_audio_path, device
        )

        # Obtener el texto transcrito con marcas de tiempo
        transcription_text = ''
        for segment in result_aligned['segments']:
            start = segment['start']
            end = segment['end']
            text = segment['text']
            transcription_text += f"[{start:.2f}s - {end:.2f}s]: {text}\n"

        transcription_results[transcription_id] = transcription_text  # Almacenar transcripción

        # Borrar archivos temporales
        os.remove(preprocessed_audio_path)
        os.remove(audio_path)

        print(f"Transcripción completada:\n{transcription_text}")
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

    # Crear un identificador único para la transcripción
    transcription_id = re.sub(r'\W+', '', os.path.splitext(normalized_filename)[0])

    # Iniciar el proceso de transcripción en un hilo separado
    thread = threading.Thread(target=transcribir_audio_async, args=(audio_path, transcription_id))
    thread.start()

    # Devolver el ID de la transcripción para que el frontend pueda hacer polling
    return jsonify({"status": "Transcripción en progreso...", "transcription_id": transcription_id}), 202

# Ruta para consultar el estado de la transcripción
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
