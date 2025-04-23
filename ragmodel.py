import os
import hashlib
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from deep_translator import GoogleTranslator
from langdetect import detect
from app_1.utils.config_reader import read_config
from django.conf import settings
from app_1.models import VectorStoreModel  # Import the model
from dotenv import load_dotenv

load_dotenv()
class DocumentProcessor:
    def __init__(self, huggingface_token):
        self.hf_token = huggingface_token
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.hf_token
        self.embeddings = self._initialize_embeddings()
        self.llm = self._initialize_llm()
        self.vector_store_dir = os.path.join(settings.BASE_DIR, 'vector_stores')
        os.makedirs(self.vector_store_dir, exist_ok=True)

    def _initialize_embeddings(self):
        model_name = "BAAI/bge-small-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

    def _initialize_llm(self):
        return HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.3",
            temperature=0.1,
            max_new_tokens=512,
            task="text-generation"
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
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        return splitter.split_text(text)

    def create_vector_store(self, texts):
        return FAISS.from_texts(texts, self.embeddings)

    def load_or_create_vector_store(self, pdf_path, store_name="default_store"):
        """
        Load existing vector store if available, otherwise create a new one
        """
        try:
            # Calculate document hash to detect changes
            doc_hash = self.get_document_hash(pdf_path)
            
            # Define the index name and path
            index_name = f"{store_name}"
            vector_store_path = os.path.join(self.vector_store_dir, index_name)
            
            # Check if we have this vector store in our database
            try:
                vector_record = VectorStoreModel.objects.get(name=store_name)
                
                # If document hasn't changed and files exist, load the existing vector store
                if vector_record.document_hash == doc_hash and os.path.exists(vector_record.vector_dir_path):
                    print(f"Loading existing vector store for {store_name}")
                    return FAISS.load_local(vector_record.vector_dir_path, self.embeddings, index_name)
                else:
                    print(f"Document changed or vector store missing, rebuilding for {store_name}")
                    return self._rebuild_and_save_vector_store(pdf_path, store_name, doc_hash, vector_store_path)
                    
            except VectorStoreModel.DoesNotExist:
                print(f"No existing vector store found for {store_name}, creating new")
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
        
        # Update or create database record
        VectorStoreModel.objects.update_or_create(
            name=store_name,
            defaults={
                'source_path': pdf_path,
                'vector_dir_path': vector_store_path,
                'document_hash': doc_hash
            }
        )
        
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

def main(pdf_path, hf_token):
    # Initialize document processor
    processor = DocumentProcessor(hf_token)

    # Load or create vector store
    vector_store = processor.load_or_create_vector_store(pdf_path)

    # Initialize QA system
    qa_system = QASystem(vector_store, processor.llm)

    return qa_system

# if __name__ == "__main__":
#     # Example usage
#     # config = read_config()
#     # HF_TOKEN = config.get("HUGGINGFACEHUB_API_TOKEN")
#     # if not HF_TOKEN:
#         # raise ValueError("HUGGINGFACEHUB_API_TOKEN is not set in the config file.")

#     PDF_PATH = os.path.join(os.path.dirname(__file__), "trainingdata/programming.pdf")
    
#     qa_system = main(PDF_PATH, HF_TOKEN)
    
#     # Example question in different languages
#     examples = [
#         {"question": "What is the role of LLM in education?", "target_lang": "en"},  # English question, English response
#         {"question": "GPTの教育における役割は何ですか？", "target_lang": "ja"},      # Japanese question, Japanese response
#         {"question": "கல்வியில் LLM இன் பங்கு என்ன?", "target_lang": "ta"},        # Tamil question, Tamil response

#     ]
    
#     for example in examples:
#         answer, source_lang = qa_system.get_answer(example["question"], target_lang=example["target_lang"])
#         print(f"\nQuestion ({source_lang}): {example['question']}")
#         print(f"Answer ({example['target_lang']}): {answer}")