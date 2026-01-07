from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
import torch

# Load the Wav2Vec2 processor and model from Hugging Face
revision_id = "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h", revision=revision_id)
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h", revision=revision_id)

def speech_to_text(audio_file, sampling_rate=16000):
    audio_file = audio_file.strip().replace("\n", "").replace("\r", "")
    # Load the audio file
    audio_input, sampling_rate_ = sf.read(audio_file)

    # Ensure the audio is at the correct sampling rate
    if sampling_rate_ != sampling_rate:
        raise ValueError(f"Audio file's sampling rate is {sampling_rate_}, but the model expects {sampling_rate} Hz.")
    # Process the audio input for the model
    inputs = processor(audio_input, return_tensors="pt", sampling_rate=sampling_rate, padding=True)

    # Perform speech-to-text
    with torch.no_grad():
        logits = model(input_values=inputs.input_values).logits

    # Decode the predicted tokens to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # Return the transcription
    return transcription.lower()

if __name__ == "__main__":
    while True:
        print("+" *50)
        audio_file = input("\033[92mEnter your WAV file path: Type 'X' or 'x' to exit: \033[0m").strip()
        if audio_file in ['X', 'x']:
            print("Exiting.")
            break
        else:
            # Transcribe the audio
            transcription = speech_to_text(audio_file)

            print("Returned Transcription:", transcription)