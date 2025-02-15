import openai
import queue
import sounddevice as sd
import numpy as np
import requests
from transformers import pipeline

# Load Hypothesis Extraction Model
hypothesis_extractor = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Load Follow-Up Question Model
followup_prompt = pipeline("text-generation", model="EleutherAI/gpt-neo-2.7B")

# Audio queue
q = queue.Queue()

# Store conversation history
conversation_history = []

# Callback function to store audio data
def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# Convert Ultron's text responses to speech using OpenAI TTS
def text_to_speech(ultron_response, output_audio_file="ultron_response.wav"):
    response = openai.Audio.create(
        model="tts-1",  # Use OpenAI's TTS model
        voice="alloy",  # Choose a voice model (e.g., alloy, echo, fable)
        input=ultron_response
    )
    
    # Save and play the audio response
    with open(output_audio_file, "wb") as f:
        f.write(response["audio"])
    
    # Play the generated speech
    sd.play(sd.read(output_audio_file)[0])
    sd.wait()
    
    # Print text response after speech
    print(ultron_response)

# Transcribe real-time audio using OpenAI Voice API
def transcribe_audio_openai(audio_file):
    with open(audio_file, "rb") as file:
        transcript = openai.Audio.transcribe("whisper-1", file)
    return transcript["text"]

# Extract and refine hypothesis from full conversation
def extract_hypothesis():
    conversation_context = "\n".join(conversation_history)
    prompt = f"Based on the following discussion, generate a rough hypothesis:\n{conversation_context}"
    result = hypothesis_extractor(prompt, max_length=100, do_sample=True)
    return result[0]['generated_text']

# Generate follow-up questions after refining hypothesis
def generate_followup(hypothesis, confidence, citation_reliability, predictions):
    prompt = f"Given the hypothesis:\n{hypothesis}\nConfidence Score: {confidence}\nCitation Reliability:\n{citation_reliability}\nPredictions:\n{predictions}\nWhat additional perspectives or clarifications can be asked to refine this idea further?"
    result = followup_prompt(prompt, max_length=100, do_sample=True)
    return result[0]['generated_text']

# Refine hypothesis after follow-up conversation
def refine_hypothesis(hypothesis):
    conversation_context = "\n".join(conversation_history)
    prompt = f"Based on the initial hypothesis:\n{hypothesis}\nand the follow-up conversation:\n{conversation_context}\nRefine the hypothesis for better clarity and scientific rigor."
    result = hypothesis_extractor(prompt, max_length=150, do_sample=True)
    return result[0]['generated_text']

# Collect user response after the follow-up question
def collect_user_response():
    text_to_speech("Your thoughts on these findings?")
    user_response = transcribe_audio_openai("temp_audio.wav")  # Capture user's voice response
    conversation_history.append(f"User: {user_response}")
    return user_response

# Query Kamiwaza for additional hypothesis validation
def query_kamiwaza(hypothesis, context):
    url = "https://kamiwaza.api/infer"  # Replace with actual API endpoint
    payload = {
        "hypothesis": hypothesis,
        "sources": ["arXiv", "PubMed", "Wikipedia", "Semantic Scholar"],  # Expanded sources
        "max_results": 100,  # Retrieve more refined and relevant results
        "context": context
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    return "No relevant insights found."

# Capture and transcribe audio
def transcribe_audio():
    samplerate = 16000  # OpenAI Whisper expects 16kHz audio
    blocksize = 1024
    channels = 1
    dtype = "float32"
    
    with sd.InputStream(samplerate=samplerate, blocksize=blocksize, channels=channels, dtype=dtype, callback=callback):
        text_to_speech("Listening for hypotheses...")
        audio_buffer = []
        while True:
            data = q.get()
            audio_buffer.append(data)
            
            # Convert to NumPy array and save as temporary audio file
            audio_np = np.concatenate(audio_buffer, axis=0)
            temp_audio_file = "temp_audio.wav"
            sd.write(temp_audio_file, audio_np, samplerate)
            
            # Transcribe using OpenAI Voice API
            transcribed_text = transcribe_audio_openai(temp_audio_file)
            conversation_history.append(f"Ultron: {transcribed_text}")
            
            # Extract hypothesis from conversation
            hypothesis = extract_hypothesis()
            
            # Collect validation metrics
            confidence, citation_reliability, predictions = query_kamiwaza(hypothesis, "Evaluate confidence score and logical consistency.")
            
            # Generate follow-up question based on validation
            followup_question = generate_followup(hypothesis, confidence, citation_reliability, predictions)
            text_to_speech(followup_question)  # Speak the follow-up question
            
            conversation_history.append(f"Ultron: {followup_question}")
            
            # Collect user response
            user_response = collect_user_response()
            conversation_history.append(f"User: {user_response}")
            
            # Refine hypothesis after follow-up conversation
            refined_hypothesis = refine_hypothesis(hypothesis)
            refined_message = f"I am checking publications to test {refined_hypothesis}."
            text_to_speech(refined_message)  # Speak refined hypothesis
            
            # Final call to Kamiwaza for inference on refined hypothesis
            final_kamiwaza_response = query_kamiwaza(refined_hypothesis, "Final validation and additional insights.")
            text_to_speech(f"Here are my findings: {final_kamiwaza_response}")  # Speak final insights

if __name__ == "__main__":
    transcribe_audio()
