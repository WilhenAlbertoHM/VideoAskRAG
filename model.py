import os
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
import whisper
from urllib.error import HTTPError
from pytubefix import YouTube
from dotenv import load_dotenv

load_dotenv()
    
model = OllamaLLM(model=os.getenv("MODEL"))
template = """
Answer the question based on the context below. 
If you can't answer the question, reply, "I don't know".

Context: {context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)


"""
Creates a transcription of the video. This is to serve as the context for the model
Parameter: video_url - the URL of the video to transcribe
Returns: None
"""
def create_transcription(video_url: str) -> None:
    video_id = video_url.split("v=")[1]
    if not os.path.exists(f"{video_id}_transcription.txt"):
        try:
            youtube = YouTube(url=video_url)
            audio = youtube.streams.filter(only_audio=True).first()
            
            # Use whisper to transcribe the audio
            whisper_model = whisper.load_model("base")
            file = audio.download(output_path=os.getcwd())
            try:
                transcription = whisper_model.transcribe(audio=file, fp16=False)["text"].strip()
                with open(f"{video_id}_transcription.txt", "w") as f:
                    f.write(transcription)
            except Exception as e:
                print(f"An error occurred: {e}")
        
        except HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
        except Exception as e:
            print(f"An error occurred: {e}")

"""
Splits the transcription into smaller Document files to feed into the model. 
Note that the model has a token limit.

Parameter: transcription - the transcription file to split
Returns: a list of Document objects containing the split transcription
"""
def split_transcription(transcription_path: str) -> list:
    if not os.path.exists(transcription_path):
        raise FileNotFoundError(f"File not found: {transcription_path}")
    
    loader = TextLoader(file_path=transcription_path)
    text_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    documents = splitter.split_documents(text_documents)
    return documents


"""
Creates the chain for the model
Parameter: documents - the list of Document objects to feed into the model
Returns: the chain
"""
def create_chain(documents: list) -> RunnableParallel:
    # Load the vector store with the documents
    embeddings = OllamaEmbeddings(model=os.getenv("MODEL"))
    vector_store = DocArrayInMemorySearch.from_documents(
        documents=documents, 
        embedding=embeddings
    )

    # Create the chain
    setup = RunnableParallel(
        context=vector_store.as_retriever(),
        question=RunnablePassthrough()
    )

    chain = setup | prompt | model 
    return chain


"""
Gets the response from the model
Parameter: input_text - the question to ask the model
Returns: the response from the model
"""
def get_response(video_url: str, input_text: str) -> str:
    # Split the transcription into smaller documents
    try:
        video_id = video_url.split("v=")[1]
        documents = split_transcription(f"{video_id}_transcription.txt")
    except FileNotFoundError as e:
        return str(e)

    # Create the chain and get the response
    chain = create_chain(documents)
    
    try:
        response = chain.invoke(input_text)
        return response
    except Exception as e:
        return str(e)
