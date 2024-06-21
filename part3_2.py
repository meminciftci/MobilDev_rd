from google.cloud import speech_v1p1beta1 as speech
from google.cloud.speech_v1p1beta1 import enums
import io

# Transcribe the given audio file
def transcribe_speech(audio_file_path):
    client = speech.SpeechClient()          # Creating a SpeechClient object

    with io.open(audio_file_path, "rb") as audio_file:  # Reading the audio file 
        content = audio_file.read()                     # Reading the audio file content

    audio = speech.types.RecognitionAudio(content=content)          # Creating a RecognitionAudio object
    config = speech.types.RecognitionConfig(                        # Creating a RecognitionConfig object
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,    # Setting the encoding to LINEAR16  
        sample_rate_hertz=16000,                    # Setting the sample rate to 16000
        language_code="tr-TR",                      # Setting the language code to Turkish
    )

    response = client.recognize(config=config, audio=audio)     # Performing the speech recognition

    for result in response.results:
        print("Transcript: {}".format(result.alternatives[0].transcript))       # Printing the transcribed text

# Replace with your audio file path
audio_file_path = "../turkish_audio.wav"    # Path to the audio file
transcribe_speech(audio_file_path)          # Transcribing the audio file
