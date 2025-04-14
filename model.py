import os
import pinecone
from langchain.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from urllib.error import HTTPError
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

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
def create_transcription(video_url: str) -> str:
    try:
        # Extract the video ID from the URL
        video_id = video_url.split("v=")[1]
        
        # Get the transcript
        transcription_list = YouTubeTranscriptApi.get_transcript(
            video_id=video_id, 
            languages=["en", "es", "fr", "de", "it"]
        )
        transcription = " ".join([item["text"] for item in transcription_list])
        
        # Split and store in Pinecone
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        texts = splitter.split_text(transcription)
        
        # Create documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "video_id": video_id,
                    "chunk_id": i,
                    "source": video_url
                }
            ) for i, chunk in enumerate(texts)
        ]
        
        # Store in Pinecone
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_query"
        )
        
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=os.getenv("PINECONE_INDEX_NAME")
        )
        
        return video_id
            
    except HTTPError as e:
        raise Exception(f"HTTP Error: {e.code} - {e.reason}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")


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
    # Load the vector store with documents using Gemini embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
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
        video_id = video_url.split("v=")[1]
        
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            task_type="retrieval_query"
        )
        
        # Initialize Pinecone vector store
        vector_store = PineconeVectorStore(
            index_name=os.getenv("PINECONE_INDEX_NAME"),
            embedding=embeddings
        )
        
        # Create retriever with metadata filter for specific video
        retriever = vector_store.as_retriever(
            search_kwargs={
                "filter": {"video_id": video_id},
                "k": 5  # Number of relevant chunks to retrieve
            }
        )
        
        # Create the chain
        setup = RunnableParallel(
            context=retriever,
            question=RunnablePassthrough()
        )
        
        chain = setup | prompt | model
        response = chain.invoke(input_text)
        return response
    
    except Exception as e:
        return str(e)
