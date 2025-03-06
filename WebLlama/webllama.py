import os
# Set the USER_AGENT environment variable to identify requests
os.environ['USER_AGENT'] = 'MyCustomUserAgent/1.0'

# Import necessary modules from langchain_community and other libraries
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.prompts import PromptTemplate
from datetime import date
from pydantic import BaseModel
from typing import Literal
import subprocess
import argparse
import logging
import ollama
import sys

def main():
    args = sys.argv[1:]  # Erste Argument (Scriptname) ignorieren
    
    if not args:
        print("Usage: webllama <command> [args...]")
        return
    
    if args[0] == "run" and len(args) > 1:
        run_model(args[1])
    else:
        subprocess.run(["ollama"] + args)

def run_model(model):
    print(f"Running model {model}") 

# Define the main ChatWeb class
class ChatWeb():
    def __init__(self):
        # Initialize the model and other attributes
        self.model, self.embeddings = self.get_model()
        self.debug = False
        self.conversation_history = []
        if self.debug:
            logging.basicConfig(level=logging.INFO)
        self.loop()
        
    # Define a Pydantic model for the query
    class Query(BaseModel):
        search_query: str
        timerange: Literal['d', 'w', 'm', 'y', 'none']

    # Define the RAG application class
    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain

        # Methode zum Ausführen der RAG-Anwendung mit Streaming
        def run(self, question, conversation_history):
            answer = ""
            # Relevante Dokumente abrufen
            documents = self.retriever.invoke(question)
            # Inhalt der abgerufenen Dokumente extrahieren
            doc_texts = "\n".join([doc.page_content for doc in documents])
            # Konversationverlauf formatieren
            history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history])
            # Aktuelles Datum im gewünschten Format
            to_date = date.today().strftime("%d.%m.%Y")
            # Eingabe für das Sprachmodell vorbereiten
            prompt_input = {"question": question, "documents": doc_texts, "history": history_text, "date": to_date}
            # Antwort des Sprachmodells streamen
            for chunk in self.rag_chain.stream(prompt_input):
                print(chunk, end="", flush=True)
                answer += chunk
            print("")
            return answer

    # Main loop to handle user input
    def loop(self):
        while True:
            self.question = input('>>> ')
            if self.question.lower() == "/help":
                print("Type /quit to exit the application.")
                continue
            if self.question.lower() == "/quit":
                sys.exit()
            self.search_query()
            wrapper = DuckDuckGoSearchAPIWrapper(time=self.time, max_results=20)
            self.search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list", num_results=20)
            self.google_search()
            if self.urls:
                self.answer_query()
            else:
                logging.error("Keine URLs gefunden.")
    
    # Method to get the model and embeddings model
    def get_model(self):
        models = ollama.list().models
        embed_model = "nomic-embed-text"
        emb_model = False
        embed_models = ["nomic-embed-text", "mxbai-embed-large", "snowflake-arctic-embed", "bge-m3", "all-minilm", "bge-large", "paraphrase-multilingual", "snowflake-arctic-embed2", "granite-embedding"]
        model = ""
        dict_models = {}

        for list_model in models:
            if list_model.model.startswith(embed_model):
                emb_model = True

        if not emb_model:
            print("No embedding model found. Downloading the embedding model.")
            ollama.pull(embed_model)

        if models:
            print("Available models:")
            i = 0
            for list_model in models:
                emb = False
                for embe_model in embed_models:
                    if list_model.model.startswith(embe_model):
                        emb = True
                        break
                if not emb:
                    print(f"{i+1} {list_model.model}")
                    dict_models[i] = list_model.model
                    i += 1

        else:
            logging.error("No models found. download a model with 'ollama run <model>'")
            sys.exit()

        while not model:
            self.model_id = input("Chose a model: ")
            try:
                model = dict_models[int(self.model_id)-1]
                for embe_model in embed_models:
                    if model.startswith(embe_model):
                        model = ""
                        continue
                        
            except Exception:
                logging.error(" Model not found.")

        ollama.chat(model=model, messages=[{"role": "user", "content": "Test"}])

        os.system("cls" if os.name == "nt" else "clear")

        return model, embed_model

    # Method to build the RAG application
    def build_rag(self):
        # List of URLs to load documents from
        docs = []

        for url in self.urls:
            try:
                docs.append(WebBaseLoader(url).load())
            except Exception:
                continue

        docs_list = [item for sublist in docs for item in sublist]
        # Initialize a text splitter with specified chunk size and overlap
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024, chunk_overlap=100
        )
        # Split the documents into chunks
        doc_splits = text_splitter.split_documents(docs_list)

        # Create embeddings for documents and store them in a vector store
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=OllamaEmbeddings(model=self.embeddings),
        )
        retriever = vectorstore.as_retriever(k=5, similarity_threshold=0.7)  # Increase k and similarity threshold

        # Define the prompt template for the LLM
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents and conversation history to answer the question.
            Answer always in the language of the question.
            Use three sentences maximum and keep the answer concise:
            Only for your context today's date: {date}, don't mention it in your answer
            Conversation History:
            {history}
            Question: {question}
            Documents: {documents}
            Answer:
            """,
            input_variables=["question", "documents", "history", "date"],
        )
        # Initialize the LLM with Llama 3.1 model
        llm = ChatOllama(
            model=self.model,
            temperature=0.7,
            num_ctx=4096,
        )
        # Create a chain combining the prompt template and LLM
        rag_chain = prompt | llm | StrOutputParser()

        # Initialize the RAG application
        rag_application = self.RAGApplication(retriever, rag_chain)
        return rag_application

    # Method to perform a Google search
    def google_search(self):
        self.urls = []
        query = self.query.replace(" ", "+")
        results = self.search.invoke(query)

        for result in results:
            self.urls.append(result['link'])
            if self.debug:
                print(result['link'])
        return self.urls

    # Method to answer the query
    def answer_query(self):
        rag_app = self.build_rag()
        answer = rag_app.run(self.question, self.conversation_history)
        self.conversation_history.append({"role": "user", "content": self.question})
        self.conversation_history.append({"role": "assistant", "content": answer})

    # Method to create a search query
    def search_query(self):
        to_date = date.today().strftime("%d.%m.%Y")
        prompt = f"""
    Todays date is {to_date}.
    Based on the following user question, formulate a concise and effective search query:
    "{self.question}"
    Your task:
    1. Create a search query of that will yield relevant results.
    2. Determine if a specific time range is needed for the search.
    Time range options:
    - 'd': Limit results to the past day. Use for very recent events or rapidly changing information.
    - 'w': Limit results to the past week. Use for recent events or topics with frequent updates.
    - 'm': Limit results to the past month. Use for relatively recent information or ongoing events.
    - 'y': Limit results to the past year. Use for annual events or information that changes yearly.
    - 'none': No time limit. Use for historical information or topics not tied to a specific time frame.
    """
        message = self.conversation_history.copy()
        message.append({"role": "user", "content": prompt})

        response = ollama.chat(
            messages=message,
            model=self.model,
            format=self.Query.model_json_schema(),
            options={"temperature": 0.5, "num_ctx": 4096}
        )

        format_query = self.Query.model_validate_json(response.message.content)
        self.query = format_query.search_query
        self.time = format_query.timerange
