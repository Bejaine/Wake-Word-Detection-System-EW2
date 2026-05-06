import serial
import time
import numpy as np
import wave
import os
import openwakeword
from openwakeword.model import Model
from gpiozero import LED

# --- NEW SDK IMPORT ---
from google import genai
from google.genai import types

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 460800
BLOCK_SIZE = 128
SYNC_HEADER = b'\xde\xad\xbe\xef'
THRESHOLD = 0.35     
DEBOUNCE_TIME = 1.5
SAMPLE_RATE  = 16000
CHUNK_SAMPLES = 1280 

# --- Hardware ---
LED_PIN = 17 
QUERY_DURATION = 5.0 # Seconds to listen after wake word
QUERY_FILE = "query.wav"

# --- Gemini API Setup ---
GOOGLE_API_KEY = "API_key"
# Initialize the modern SDK Client
client = genai.Client(api_key=GOOGLE_API_KEY)
# ---------------------

def save_wav(filename, audio_data, sample_rate=16000):
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

def ask_gemini(audio_file_path):
    """Uploads the raw audio file to Gemini using the modern SDK."""
    print("\n[API] Uploading audio to Cloud...")
    try:
        # New SDK File Upload API
        audio_file = client.files.upload(file=audio_file_path)
        prompt = "Listen to this audio command and respond directly and concisely. Do not use markdown."
        
        # New SDK Generate Content API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, audio_file]
        )
        return response.text
    except Exception as e:
        return f"Error 404: {e}"

def record_query(ser, duration_sec, sample_rate=16000):
    target_samples = duration_sec * sample_rate
    collected = np.array([], dtype=np.int16)
    serial_buffer = bytearray()
    
    print(f"[!] Recording query for {duration_sec} seconds...")

    while len(collected) < target_samples:
        if ser.in_waiting > 0:
            serial_buffer.extend(ser.read(ser.in_waiting))

        header_pos = serial_buffer.find(SYNC_HEADER)
        if header_pos != -1:
            block_end = header_pos + 4 + (BLOCK_SIZE * 2) 
            if len(serial_buffer) >= block_end:
                raw_audio = serial_buffer[header_pos + 4 : block_end]
                serial_buffer = serial_buffer[block_end:]
                
                # Convert raw bytes directly to int16 array
                chunk = np.frombuffer(raw_audio, dtype=np.int16)
                collected = np.concatenate((collected, chunk))

    save_wav(QUERY_FILE, collected[:int(target_samples)], sample_rate)

def run_assistant_pipeline():
    print("Initializing Wake Word Engine...")
    all_paths = openwakeword.get_pretrained_model_paths()
    alexa_path = next((p for p in all_paths if "alexa" in p.lower()), None)
    oww_model = Model(wakeword_model_paths=[alexa_path])
    model_key = list(oww_model.models.keys())[0]
    
    status_led = LED(LED_PIN)
    status_led.off()

    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.reset_input_buffer()
        serial_buffer = bytearray()
        audio_accumulator = np.array([], dtype=np.int16)
        last_detection_time = 0.0

        print("\n=== Assistant Online. Say 'Alexa' to begin. ===")

        while True:
            if ser.in_waiting > 0:
                serial_buffer.extend(ser.read(ser.in_waiting))

            header_pos = serial_buffer.find(SYNC_HEADER)
            if header_pos != -1:
                block_end = header_pos + 4 + (BLOCK_SIZE * 2)

                if len(serial_buffer) >= block_end:
                    raw_audio = serial_buffer[header_pos + 4 : block_end]
                    serial_buffer = serial_buffer[block_end:]

                    # Extract raw chunk and append directly
                    chunk = np.frombuffer(raw_audio, dtype=np.int16)
                    audio_accumulator = np.concatenate((audio_accumulator, chunk))

                    if len(audio_accumulator) >= CHUNK_SAMPLES:
                        prediction = oww_model.predict(audio_accumulator[:CHUNK_SAMPLES])
                        score = prediction[model_key]
                        audio_accumulator = audio_accumulator[CHUNK_SAMPLES:]

                        if score > THRESHOLD:
                            current_time = time.time()
                            if (current_time - last_detection_time) > DEBOUNCE_TIME:
                                print(f"\n[!!!] WAKE WORD DETECTED (Score: {score:.3f})")
                                
                                status_led.on()
                                record_query(ser, QUERY_DURATION, SAMPLE_RATE)
                                status_led.off()
                                
                                answer = ask_gemini(QUERY_FILE)
                                print(f"\nAlexa Says:\n{answer}\n")
                                
                                ser.reset_input_buffer()
                                serial_buffer.clear()
                                audio_accumulator = np.array([], dtype=np.int16)
                                last_detection_time = time.time()
                                print("=== Listening for Wake Word ===")

    except serial.SerialException as e:
        print(f"Serial Hardware Error: {e}")
    except KeyboardInterrupt:
        print("\nPipeline terminated by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    run_assistant_pipeline()