import os
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama.embeddings import OllamaEmbeddings
from urllib.error import HTTPError
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()
    
model = OllamaLLM(model=os.getenv("MODEL"))
template = """
You are a friendly and knowledgeable AI assistant that answers questions about a video based on the provided transcript. 
Refer to the 'context' or 'transcript' as 'video' to avoid confusion, as the user only sees the video and not a 'video transcript'. 
Respond with enthusiasm, clarity, and conciseness, keeping your answers directly relevant to the question. 
If the video does not contain the answer, kindly say so. 
Match the language of your response to the language of the question.

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
    try:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[1]
        transcription_file = f"{video_id}_transcription.txt"
        
        # If transcription file does not exist, create it and write it to a .txt file
        if not os.path.exists(transcription_file):
            transcription_list = YouTubeTranscriptApi.get_transcript(
                video_id=video_id, 
                languages=["en", "es", "fr", "de", "it"]
            )
            transcription = " ".join([item["text"] for item in transcription_list])

            with open(transcription_file, "w") as f:
                f.write(transcription)
            
    except HTTPError as e:
        print(f"HTTP Error: {e.code} - {e.reason}")
    except FileNotFoundError as e:
        print(f"File not found: {e}")
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
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
    vector_store = PineconeVectorStore.from_documents(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
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
    try:
        # Split the transcription into smaller documents
        video_id = video_url.split("v=")[1]
        documents = split_transcription(f"{video_id}_transcription.txt")

        # Create the chain and return the response
        chain = create_chain(documents)
        response = chain.invoke(input_text)
        return response
    
    except FileNotFoundError as e:
        return str(e)
    except Exception as e:
        return str(e)
