import os
import hashlib
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator
from langdetect import detect
from dotenv import load_dotenv
import tempfile
import uuid

load_dotenv()

class DocumentProcessor:
    def __init__(self, groq_api_key=None):
        self.api_key = groq_api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables or provided as parameter")
        
        os.environ["GROQ_API_KEY"] = self.api_key
        self.embeddings = self._initialize_embeddings()
        self.llm = self._initialize_llm()
        
        # Create temporary directory for vector stores when not using Django
        # Create vector_stores directory in the same directory as the script
        self.vector_store_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vector_stores')
        os.makedirs(self.vector_store_dir, exist_ok=True)
        print(f"Vector stores will be saved in: {self.vector_store_dir}")

    def _initialize_embeddings(self):
        model_name = "BAAI/bge-small-en-v1.5"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def _initialize_llm(self):
        return ChatGroq(
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

    def get_document_hash(self, file_path):
        """Calculate MD5 hash of a file to detect changes"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def process_pdf(self, pdf_path):
        reader = PdfReader(pdf_path)
        raw_text = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                raw_text += text
        return raw_text

    def process_text(self, text):
        # Using RecursiveCharacterTextSplitter instead of CharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", " ", ""],
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return splitter.split_text(text)

    def create_vector_store(self, texts):
        return FAISS.from_texts(texts, self.embeddings)

    def load_or_create_vector_store(self, pdf_path, store_name=None):
        """
        Load existing vector store if available, otherwise create a new one
        """
        try:
            # Calculate document hash to detect changes
            doc_hash = self.get_document_hash(pdf_path)
            
            # Generate a store name if not provided
            if not store_name:
                store_name = f"store_{os.path.basename(pdf_path)}_{doc_hash[:8]}"
            
            # Define the index name and path
            index_name = f"{store_name}"
            vector_store_path = os.path.join(self.vector_store_dir, index_name)
            
            # Check if vector store exists
            if os.path.exists(vector_store_path):
                try:
                    print(f"Loading existing vector store for {store_name}")
                    return FAISS.load_local(vector_store_path, self.embeddings, index_name)
                except Exception as e:
                    print(f"Error loading vector store: {str(e)}")
                    print("Creating new vector store")
            
            # If not exists or error loading, rebuild
            return self._rebuild_and_save_vector_store(pdf_path, store_name, doc_hash, vector_store_path)
                
        except Exception as e:
            print(f"Error in load_or_create_vector_store: {str(e)}")
            # If any error occurs, fall back to creating a new vector store
            raw_text = self.process_pdf(pdf_path)
            texts = self.process_text(raw_text)
            return self.create_vector_store(texts)

    def _rebuild_and_save_vector_store(self, pdf_path, store_name, doc_hash, vector_store_path):
        """Helper method to rebuild and save vector store"""
        # Process the PDF and create vector store
        raw_text = self.process_pdf(pdf_path)
        texts = self.process_text(raw_text)
        vector_store = self.create_vector_store(texts)
        
        # Ensure directory exists
        os.makedirs(vector_store_path, exist_ok=True)
        
        # Save the vector store
        vector_store.save_local(vector_store_path, index_name=store_name)
        
        # Save metadata in a simple file since we're not using Django
        with open(os.path.join(vector_store_path, "metadata.txt"), "w") as f:
            f.write(f"source_path: {pdf_path}\n")
            f.write(f"document_hash: {doc_hash}\n")
        
        return vector_store

class QASystem:
    def __init__(self, vector_store, llm):
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True
        )

    def get_answer(self, question, target_lang=None):
        """
        Get answer to a question in the specified target language.
        
        Args:
            question (str): The question in any language
            target_lang (str, optional): ISO language code for the response language.
                                        If None, returns answer in English.
        
        Returns:
            tuple: (answer, detected_source_language)
        """
        # Detect input language
        source_lang = detect(question)
        
        # Translate question to English if needed
        if source_lang != 'en':
            translator = GoogleTranslator(source='auto', target='en')
            question = translator.translate(question)

        # Get answer in English
        answer = self.qa_chain.invoke(question)["result"]

        # Translate answer if needed
        if target_lang and target_lang != 'en':
            translator = GoogleTranslator(source='en', target=target_lang)
            answer = translator.translate(answer)

        return answer, source_lang

def main(pdf_path, groq_api_key=None):
    # Initialize document processor
    processor = DocumentProcessor(groq_api_key)

    # Load or create vector store
    vector_store = processor.load_or_create_vector_store(pdf_path)

    # Initialize QA system
    qa_system = QASystem(vector_store, processor.llm)

    return qa_system

if __name__ == "__main__":
    # If PDF_PATH is not provided, you can set it here
    PDF_PATH = input("Enter the path to your PDF file: ")
    
    # Initialize the QA system
    qa_system = main(PDF_PATH)
    
    print("\nEnter your questions (type 'exit' to quit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'exit':
            break
            
        target_lang = input("Target language code (leave empty for English): ")
        if not target_lang:
            target_lang = 'en'
            
        answer, source_lang = qa_system.get_answer(question, target_lang=target_lang)
        print(f"\nDetected source language: {source_lang}")
        print(f"Answer ({target_lang}): {answer}")