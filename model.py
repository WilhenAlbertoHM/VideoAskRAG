import os
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import VectorStore
from langchain_community.document_loaders import TextLoader
import tempfile
import whisper
from pytube import YouTube


model = OllamaLLM(model="llama3.2")
template = """
Answer the question based on the context below. 
If you can't answer the question, reply, "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

"""
Creates a transcription of the video. This is to serve as the context for the model
Parameter: video_url - the URL of the video to transcribe
Returns: None
"""
def get_transcription(video_url: str):
    if not os.path.exists("transcription.txt"):
        youtube = YouTube(video_url)
        audio = youtube.streams.filter(only_audio=True).first()
        
        # Use whisper to transcribe the audio
        whisper_model = whisper.load_model("base")
        with tempfile.TemporaryDirectory() as temp_dir:
            file = audio.download(output_path=temp_dir)
            transcription = whisper_model.transcribe(file, fp16=False)["text"].strip()
            
            # Save the transcription to a file
            with open("transcription.txt", "w") as f:
                f.write(transcription)



"""
Gets the response from the model
Parameter: input_text - the question to ask the model
Returns: the response from the model
"""
def get_response(input_text: str):
    pass

