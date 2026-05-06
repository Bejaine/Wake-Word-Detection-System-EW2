import serial
import time
import numpy as np
import wave
import openwakeword
from openwakeword.model import Model

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 460800
BLOCK_SIZE = 128
SYNC_HEADER = b'\xde\xad\xbe\xef'
THRESHOLD = 0.35    
DEBOUNCE_TIME = 1.5
SAMPLE_RATE  = 16000
CHUNK_SAMPLES = 1280 # Rigid 80ms frames required by openWakeWord

# --- Debug Recording ---
DEBUG_RECORD = True
DEBUG_DURATION_SEC = 5
DEBUG_OUTPUT_FILE = "debug_capture_raw.wav"
# ---------------------

def save_wav(filename, audio_data, sample_rate=16000):
    """Saves raw audio data to a WAV file for debugging."""
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    db = 20 * np.log10(rms / 32768) if rms > 0 else -999
    
    print(f"\n[DEBUG] Saved '{filename}'")
    print(f"[DEBUG]   Level    : {db:.1f} dBFS")

def collect_debug_audio(ser, duration_sec, sample_rate=16000):
    """Collects raw audio from the serial port for a specific duration."""
    target_samples = duration_sec * sample_rate
    collected = np.array([], dtype=np.int16)
    serial_buffer = bytearray()
    
    print(f"\n[DEBUG] Recording {duration_sec}s... say 'Alexa' a few times now!")

    while len(collected) < target_samples:
        if ser.in_waiting > 0:
            serial_buffer.extend(ser.read(ser.in_waiting))

        header_pos = serial_buffer.find(SYNC_HEADER)
        if header_pos != -1:
            block_end = header_pos + 4 + (BLOCK_SIZE * 2) 
            if len(serial_buffer) >= block_end:
                raw_audio = serial_buffer[header_pos + 4 : block_end]
                serial_buffer = serial_buffer[block_end:]
                
                # Convert raw bytes directly to int16 array without filtering
                chunk = np.frombuffer(raw_audio, dtype=np.int16)
                collected = np.concatenate((collected, chunk))

        elapsed = len(collected) / sample_rate
        print(f"\r[DEBUG] Captured: {elapsed:.1f}s / {duration_sec}s", end='', flush=True)

    return collected[:target_samples]

def run_wake_word_pipeline():
    print("Initializing openWakeWord Model for 'Alexa'...")

    all_paths = openwakeword.get_pretrained_model_paths()
    alexa_path = next((p for p in all_paths if "alexa" in p.lower()), None)

    if alexa_path is None:
        print("ERROR: No alexa model found.")
        return

    oww_model = Model(wakeword_model_paths=[alexa_path])
    model_key = list(oww_model.models.keys())[0]
    
    print(f"Model loaded. Prediction key: '{model_key}'")
    print("DSP bypassed: Feeding raw ESP32 audio directly to model.")

    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.reset_input_buffer()

        if DEBUG_RECORD:
            debug_audio = collect_debug_audio(ser, DEBUG_DURATION_SEC, SAMPLE_RATE)
            save_wav(DEBUG_OUTPUT_FILE, debug_audio, SAMPLE_RATE)
            print("\n[DEBUG] Entering real-time detection...\n")
            ser.reset_input_buffer()

        serial_buffer = bytearray()
        audio_accumulator = np.array([], dtype=np.int16)
        last_detection_time = 0.0
        
        print("Listening continuously... (Press Ctrl+C to exit)")

        while True:
            if ser.in_waiting > 0:
                serial_buffer.extend(ser.read(ser.in_waiting))

            header_pos = serial_buffer.find(SYNC_HEADER)

            if header_pos != -1:
                block_end = header_pos + 4 + (BLOCK_SIZE * 2)

                if len(serial_buffer) >= block_end:
                    raw_audio = serial_buffer[header_pos + 4 : block_end]
                    serial_buffer = serial_buffer[block_end:]

                    # Extract raw chunk and append to accumulator
                    chunk = np.frombuffer(raw_audio, dtype=np.int16)
                    audio_accumulator = np.concatenate((audio_accumulator, chunk))

                    # Process when enough samples are accumulated
                    if len(audio_accumulator) >= CHUNK_SAMPLES:
                        # Feed the 1280-sample array directly into the model
                        prediction = oww_model.predict(audio_accumulator[:CHUNK_SAMPLES])
                        score = prediction[model_key]

                        # Remove the processed chunk from the accumulator
                        audio_accumulator = audio_accumulator[CHUNK_SAMPLES:]

                        print(f"  Score: {score:.4f}    ", end='\r')

                        if score > THRESHOLD:
                            current_time = time.time()
                            if (current_time - last_detection_time) > DEBOUNCE_TIME:
                                print(f"\n\n[!!!] WAKE WORD DETECTED: Alexa! (Confidence: {score:.3f})\n")
                                last_detection_time = current_time

    except serial.SerialException as e:
        print(f"Serial Hardware Error: {e}")
    except KeyboardInterrupt:
        print("\nPipeline terminated by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    run_wake_word_pipeline()import serial
import time
import numpy as np
import wave
import openwakeword
from openwakeword.model import Model

# --- Configuration ---
SERIAL_PORT = '/dev/ttyUSB0'
BAUD_RATE = 460800
BLOCK_SIZE = 128
SYNC_HEADER = b'\xde\xad\xbe\xef'
THRESHOLD = 0.35    
DEBOUNCE_TIME = 1.5
SAMPLE_RATE  = 16000
CHUNK_SAMPLES = 1280 # Rigid 80ms frames required by openWakeWord

# --- Debug Recording ---
DEBUG_RECORD = True
DEBUG_DURATION_SEC = 5
DEBUG_OUTPUT_FILE = "debug_capture_raw.wav"
# ---------------------

def save_wav(filename, audio_data, sample_rate=16000):
    """Saves raw audio data to a WAV file for debugging."""
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    rms = np.sqrt(np.mean(audio_data.astype(np.float64)**2))
    db = 20 * np.log10(rms / 32768) if rms > 0 else -999
    
    print(f"\n[DEBUG] Saved '{filename}'")
    print(f"[DEBUG]   Level    : {db:.1f} dBFS")

def collect_debug_audio(ser, duration_sec, sample_rate=16000):
    """Collects raw audio from the serial port for a specific duration."""
    target_samples = duration_sec * sample_rate
    collected = np.array([], dtype=np.int16)
    serial_buffer = bytearray()
    
    print(f"\n[DEBUG] Recording {duration_sec}s... say 'Alexa' a few times now!")

    while len(collected) < target_samples:
        if ser.in_waiting > 0:
            serial_buffer.extend(ser.read(ser.in_waiting))

        header_pos = serial_buffer.find(SYNC_HEADER)
        if header_pos != -1:
            block_end = header_pos + 4 + (BLOCK_SIZE * 2) 
            if len(serial_buffer) >= block_end:
                raw_audio = serial_buffer[header_pos + 4 : block_end]
                serial_buffer = serial_buffer[block_end:]
                
                # Convert raw bytes directly to int16 array without filtering
                chunk = np.frombuffer(raw_audio, dtype=np.int16)
                collected = np.concatenate((collected, chunk))

        elapsed = len(collected) / sample_rate
        print(f"\r[DEBUG] Captured: {elapsed:.1f}s / {duration_sec}s", end='', flush=True)

    return collected[:target_samples]

def run_wake_word_pipeline():
    print("Initializing openWakeWord Model for 'Alexa'...")

    all_paths = openwakeword.get_pretrained_model_paths()
    alexa_path = next((p for p in all_paths if "alexa" in p.lower()), None)

    if alexa_path is None:
        print("ERROR: No alexa model found.")
        return

    oww_model = Model(wakeword_model_paths=[alexa_path])
    model_key = list(oww_model.models.keys())[0]
    
    print(f"Model loaded. Prediction key: '{model_key}'")
    print("DSP bypassed: Feeding raw ESP32 audio directly to model.")

    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        ser.reset_input_buffer()

        if DEBUG_RECORD:
            debug_audio = collect_debug_audio(ser, DEBUG_DURATION_SEC, SAMPLE_RATE)
            save_wav(DEBUG_OUTPUT_FILE, debug_audio, SAMPLE_RATE)
            print("\n[DEBUG] Entering real-time detection...\n")
            ser.reset_input_buffer()

        serial_buffer = bytearray()
        audio_accumulator = np.array([], dtype=np.int16)
        last_detection_time = 0.0
        
        print("Listening continuously... (Press Ctrl+C to exit)")

        while True:
            if ser.in_waiting > 0:
                serial_buffer.extend(ser.read(ser.in_waiting))

            header_pos = serial_buffer.find(SYNC_HEADER)

            if header_pos != -1:
                block_end = header_pos + 4 + (BLOCK_SIZE * 2)

                if len(serial_buffer) >= block_end:
                    raw_audio = serial_buffer[header_pos + 4 : block_end]
                    serial_buffer = serial_buffer[block_end:]

                    # Extract raw chunk and append to accumulator
                    chunk = np.frombuffer(raw_audio, dtype=np.int16)
                    audio_accumulator = np.concatenate((audio_accumulator, chunk))

                    # Process when enough samples are accumulated
                    if len(audio_accumulator) >= CHUNK_SAMPLES:
                        # Feed the 1280-sample array directly into the model
                        prediction = oww_model.predict(audio_accumulator[:CHUNK_SAMPLES])
                        score = prediction[model_key]

                        # Remove the processed chunk from the accumulator
                        audio_accumulator = audio_accumulator[CHUNK_SAMPLES:]

                        print(f"  Score: {score:.4f}    ", end='\r')

                        if score > THRESHOLD:
                            current_time = time.time()
                            if (current_time - last_detection_time) > DEBOUNCE_TIME:
                                print(f"\n\n[!!!] WAKE WORD DETECTED: Alexa! (Confidence: {score:.3f})\n")
                                last_detection_time = current_time

    except serial.SerialException as e:
        print(f"Serial Hardware Error: {e}")
    except KeyboardInterrupt:
        print("\nPipeline terminated by user.")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

if __name__ == "__main__":
    run_wake_word_pipeline()