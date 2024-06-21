import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-turkish")    # Loading the processor for Turkish
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-turkish")           # Loading the pre-trained model for Turkish

audio_input, sample_rate = sf.read("turkish_audio.wav")                             # Loading the audio file
inputs = processor(audio_input, sampling_rate=sample_rate, return_tensors="pt", padding=True)   # Preprocessing the audio file

# Performing inference using the pre-trained model and the preprocessed audio file
with torch.no_grad():
    logits = model(inputs.input_values).logits      

# Decoding the predicted IDs to text
predicted_ids = torch.argmax(logits, dim=-1)            # Decoding the predicted IDs
transcription = processor.batch_decode(predicted_ids)   # Decoding the predicted IDs to text
print(transcription[0])                                 # Printing the transcribed text
