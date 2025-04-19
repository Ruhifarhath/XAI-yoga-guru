from transformers import pipeline
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline

# Function to build the full RAG chain
def get_rag_chain():
    # Load text knowledge base
    loader = TextLoader("yoga_knowledge.txt")
    documents = loader.load()

    # Split into manageable chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(documents)

    # Create embeddings (using Hugging Face embeddings)
    embeddings = HuggingFaceEmbeddings()

    # Create FAISS vector store
    vectordb = FAISS.from_documents(docs, embeddings)

    # Setup local Hugging Face model pipeline (e.g., flan-t5)
    pipe = pipeline("text2text-generation", model="google/flan-t5-base", tokenizer="google/flan-t5-base")
    llm = HuggingFacePipeline(pipeline=pipe)

    # Build RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=False
    )
    
    return qa_chain
