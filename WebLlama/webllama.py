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
import importlib.metadata
import subprocess
import logging
import ollama
import sys
import re

def main():
    args = sys.argv[1:]  # Ignore the first argument (script name)
    
    if not args:
        print("""Usage:
  webllama [flags]
  webllama [command]

Available Commands:
  serve       Start ollama
  create      Create a model from a Modelfile
  show        Show information for a model
  run         Run a model
  stop        Stop a running model
  pull        Pull a model from a registry
  push        Push a model to a registry
  list        List models
  ps          List running models
  cp          Copy a model
  rm          Remove a model
  help        Help about any command

Flags:
  -h, --help      help for webllama
  -v, --version   Show version information

Use "webllama [command] --help" for more information about a command.\n""")
        return
    
    if args[0] == "run" and len(args) > 1:
        WebLlama(args[1], args[2:])
    elif args[0] == "--version":
        version = importlib.metadata.version("WebLlama")
        print(f"webllama version is {version}")
        subprocess.run(["ollama"] + ["--version"])
    else:
        subprocess.run(["ollama"] + args)

# Define the main ChatWeb class
class WebLlama():
    def __init__(self, model, flags):
        # Initialize the model and other attributes
        self.model = model
        self.embeddings = "paraphrase-multilingual"
        self.debug = False
        self.format = None
        self.history = True
        self.verbose = False
        self.seed = None
        self.predict = None
        self.top_k = None
        self.top_p = None
        # self.min_p = None
        self.num_ctx = 4096
        self.temperature = None
        self.repeat_penalty = None
        self.repeat_last_n = None
        self.num_gpu = None
        self.stop = None
        self.num_results = 15
        self.given_results = 5
        self.websearch_on = True
        self.conversation_history = []
        self.keep_alive = None
        if flags:
            if ("-h" or "--help") in flags:
                print("""Run a model

Usage:
  ollama run MODEL [PROMPT] [flags]

Flags:
      --format string      Response format (e.g. json)
  -h, --help               help for run
      --keepalive string   Duration to keep a model loaded (e.g. 5m)
      --verbose            Show timings for response
      --noweb              Disables the websearch
      --debug              Enables debug mode

Environment Variables:
      OLLAMA_HOST                IP Address for the ollama server (default 127.0.0.1:11434)
      OLLAMA_NOHISTORY           Do not preserve readline history\n""")
            else:
                if "--verbose" in flags:
                    self.verbose = True
                if "--debug" in flags:
                    self.debug = True
                if "--format" in flags:
                    index = flags.index("--format")
                    self.format = flags[index+1]
                if "--keepalive" in flags:
                    index = flags.index("--keepalive")
                    keepalive_str = flags[index + 1]
                    # Prüfe, ob das Format exakt "<Zahl><Einzelbuchstabe>" ist
                    m = re.fullmatch(r'(\d+)([a-zA-Z]+)', keepalive_str)
                    if m:
                        number, unit = m.groups()
                        if unit in ["s", "m", "h"]:
                            self.keep_alive = keepalive_str
                        else:
                            print(f'Error: time: unknown unit "{unit}" in duration "{keepalive_str}"')
                            sys.exit()
                    else:
                        # Prüfe auf das Format mit zusätzlichen Ziffern, z.B. "55s5" oder "55zt5"
                        m2 = re.fullmatch(r'(\d+)([a-zA-Z]+)(\d+)$', keepalive_str)
                        if m2:
                            number, unit, extra = m2.groups()
                            if unit in ["s", "m", "h"]:
                                print(f'Error: time: missing unit in duration "{keepalive_str}"')
                            else:
                                print(f'Error: time: unknown unit "{unit}" in duration "{keepalive_str}"')
                            sys.exit()
                        else:
                            print(f'Error: time: invalid duration "{keepalive_str}"')
                            sys.exit()

                if "--noweb" in flags:
                    self.websearch_on = False

        self.get_model(model)

        if self.debug:
            logging.basicConfig(level=logging.INFO)
        self.loop()
        
    # Define a Pydantic model for the query
    class Query(BaseModel):
        search_query: str | None
        timerange: Literal['d', 'w', 'm', 'y', 'none']

    class Websearch(BaseModel):
        websearch: bool

    # Define the RAG application class
    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain

        # Method to run the RAG application with streaming
        def run(self, question, conversation_history):
            answer = ""
            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            # Extract content from the retrieved documents
            doc_texts = "\n".join([doc.page_content for doc in documents])
            # Format conversation history
            history_text = "\n".join([f"{entry['role']}: {entry['content']}" for entry in conversation_history]) if conversation_history else ""
            # Current date in the desired format
            to_date = date.today().strftime("%d.%m.%Y")
            # Prepare input for the language model
            prompt_input = {"question": question, "documents": doc_texts, "history": history_text, "date": to_date}
            # Stream the language model's response
            answer = self.rag_chain.stream(prompt_input)
            return answer

    # Main loop to handle user input
    def loop(self):
        while True:
            try:
                self.question = self.multiline_input()
            except (KeyboardInterrupt, EOFError):
                print("")
                sys.exit()
            
            if self.question.startswith("/"):
                self.commands()
                continue

            if self.websearch_on:
                prompt = f"""
Today's date is {date.today().strftime("%d.%m.%Y")}.

Task: Determine whether additional context from internet sources is required to answer the user's question.  

**Instructions:**  
1. Carefully analyze the user's question and the chat history.  
2. Ignore any pre-existing knowledge and questions in the chat history, concentrate only on the users question.  
3. Respond with **'True'** if the question requires **external, real-time, or highly specific data from the Internet** or if your last knowledge update is too old to answer the question accurately. Examples:  
   - Recent news, weather, stock prices, sports results, events.  
   - Current product prices, availability, schedules, or policies.  
   - Information about specific people, locations, or businesses.
   - Political events, election results, or government policies.
4. Respond with **'False'** if:  
   - The question can be answered based on the given chat history.
   - The question is conversational (e.g., greetings, small talk, "Thank you").  
   - The question addresses you.
5. If you are unsure about the answer, choose **'True'**.

**User question:** "{self.question}"  
"""

                if self.history:
                    convo = self.conversation_history.copy()
                    convo.append({"role": "user", "content": prompt})
                response = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=False, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=0.5, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(convo if self.history else prompt)
                self.websearch = self.Websearch.model_validate_json(response.content).websearch
                if self.debug:
                    print(self.websearch)

            try:
                if self.websearch:
                    self.search_query()
                    if self.query == None or self.query == 'None':
                        if self.history:
                            convo = self.conversation_history.copy()
                            prompt = f"""
    You are an AI assistant named **WebLlama**. Your task is to process the user's input **without modifying, correcting, or altering factual statements**, even if they appear incorrect.  
    Only for your context: today's date is {date.today().strftime("%d.%m.%Y")}.

    ### **Instructions:**  
    1. **Do NOT correct, fact-check, or modify** any user statements, even if they contain apparent errors.  
    2. **Only perform the requested task** (e.g., translation, summarization, formatting), without adding comments, opinions, or corrections.  
    3. If explicitly asked to correct something, then and only then should you provide corrections.  
    4. Respond **neutrally and objectively**, without assuming that the user wants fact-checking.  
    """
                            convo.insert(0, {"role": "system", "content": prompt})
                            convo.append({"role": "user", "content": self.question})
                        self.answer = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.format, verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).stream(convo if self.history else self.question)
                        full_answer = ""
                        print(" " * 30, end="\r")
                        for chunk in self.answer:
                            print(chunk.content, end="", flush=True)
                            full_answer += chunk.content
                        print("\n")
                    else:
                        print("Performing web search...", end="\r")
                        wrapper = DuckDuckGoSearchAPIWrapper(time=self.time, max_results=self.num_results)
                        self.search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list", num_results=self.num_results)
                        self.ddg_search()
                        if self.urls:
                            self.answer_query()
                        else:
                            logging.error("Keine URLs gefunden.")
                else:
                    # answer = ollama.chat(model=self.model, messages=self.conversation_history, stream=True, format=self.format)
                    if self.history:
                        convo = self.conversation_history.copy()
                        prompt = f"""
You are an AI assistant named **WebLlama**. Your task is to process the user's input **without modifying, correcting, or altering factual statements**, even if they appear incorrect.  
Only for your context: today's date is {date.today().strftime("%d.%m.%Y")}.

### **Instructions:**  
1. **Do NOT correct, fact-check, or modify** any user statements, even if they contain apparent errors.  
2. **Only perform the requested task** (e.g., translation, summarization, formatting), without adding comments, opinions, or corrections.  
3. If explicitly asked to correct something, then and only then should you provide corrections.  
4. Respond **neutrally and objectively**, without assuming that the user wants fact-checking.  
"""
                        convo.insert(0, {"role": "system", "content": prompt})
                        convo.append({"role": "user", "content": self.question})
                    full_answer = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.format, verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(convo if self.history else self.question)
                    if self.debug:
                        print(full_answer.content)
                    # full_answer = ""
                    # print(" " * 30, end="\r")
                    # for chunk in self.answer:
                    #     print(chunk.content, end="", flush=True)
                    #     full_answer += chunk.content
                    # print("\n")
                    f"""
Today's date is {date.today().strftime("%d.%m.%Y")}.

Task: Determine whether additional context from internet sources is required to answer the user's question based on the provided answer.

**Instructions:**  
1. Analyze the question and the given answer carefully.  
2. Respond with **'True'** if the provided answer requires **external, real-time, or highly specific data from the Internet**. Examples:  
   - The answer is incorrect or uncertain.  
   - There is no clear answer.  
   - The question or the question refers to real-time information (e.g., news, weather, stock prices).
   - The question needs any context to answer.  
3. Respond with **'False'** if:  
   - The answer is complete, correct, and does not require internet research.  
   - The question is about general knowledge (e.g., math, history, defined concepts).  
   - The question contains polite phrases, small talk, or expressions of gratitude.  
   - The question is directly addressing you (e.g., "How are you?").  

### Examples:
   - "What is the capital of France?" → False  
   - "What is the weather like today?" → True  
   - "Thanks for your response!" → False  

**User question:** "{self.question}"  
**Provided answer:** "{full_answer}"  
"""

                    response = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=False, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=0.5, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(convo if self.history else prompt)
                    self.websearch = self.Websearch.model_validate_json(response.content).websearch
                    if self.debug:
                        print(self.websearch)
                    if self.websearch:
                        self.search_query()
                        if self.query == None or self.query == 'None':
                            if self.history:
                                convo = self.conversation_history.copy()
                                prompt = f"""
        You are an AI assistant named **WebLlama**. Your task is to process the user's input **without modifying, correcting, or altering factual statements**, even if they appear incorrect.  
        Only for your context: today's date is {date.today().strftime("%d.%m.%Y")}.

        ### **Instructions:**  
        1. **Do NOT correct, fact-check, or modify** any user statements, even if they contain apparent errors.  
        2. **Only perform the requested task** (e.g., translation, summarization, formatting), without adding comments, opinions, or corrections.  
        3. If explicitly asked to correct something, then and only then should you provide corrections.  
        4. Respond **neutrally and objectively**, without assuming that the user wants fact-checking.  
        """
                                convo.insert(0, {"role": "system", "content": prompt})
                                convo.append({"role": "user", "content": self.question})
                            self.answer = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.format, verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).stream(convo if self.history else self.question)
                            full_answer = ""
                            print(" " * 30, end="\r")
                            for chunk in self.answer:
                                print(chunk.content, end="", flush=True)
                                full_answer += chunk.content
                            print("\n")
                        else:
                            print("Performing web search...", end="\r")
                            wrapper = DuckDuckGoSearchAPIWrapper(time=self.time, max_results=self.num_results)
                            self.search = DuckDuckGoSearchResults(api_wrapper=wrapper, output_format="list", num_results=self.num_results)
                            self.ddg_search()
                            if self.urls:
                                self.answer_query()
                            else:
                                logging.error("Keine URLs gefunden.")
                    else:
                        print(full_answer.content)
                        # print("\n")
                        if self.history:
                            self.conversation_history.append({"role": "user", "content": self.question})
                            self.conversation_history.append({"role": "assistant", "content": full_answer})
            except KeyboardInterrupt:
                self.answer = None
                print("\n")
    
    # Method to get the model and embeddings model
    def get_model(self, model):
        try:
            emb = OllamaEmbeddings(model=self.embeddings)
            emb.embed_query("test")
        except ollama.ResponseError:
            try:
                subprocess.run(["ollama"] + ["pull"] + [self.embeddings])
                emb = OllamaEmbeddings(model=self.embeddings)
                emb.embed_query("test")
            except ollama.ResponseError:
                print(f"Error: model '{self.embeddings}' not found\n")
                sys.exit()
        try:
            ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke("test")
        except ollama.ResponseError:
            try:
                subprocess.run(["ollama"] + ["pull"] + [self.model])
                ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke("test")
            except ollama.ResponseError:
                print(f"Error: model '{self.model}' not found\n")
                sys.exit()

    def multiline_input(self, prompt=">>> "):
        first_line = input(prompt)  # Get the first line of input

        # Check if the input starts with """
        if (first_line.strip().startswith('"""')):
            print("Multiline input detected. Enter more lines (finish with \"\"\"):")
            lines = []  # List for storing lines (excluding the first """)

            while True:
                line = input()  # Read more lines
                if line.strip() == '"""':  # Stop if a line only contains """
                    break
                lines.append(line)

            return "\n".join(lines)  # Return the multiline input as a single string

        return first_line  # Return normal input if it doesn't start with """
    
    def extract_between_system_and_parameter(self, text):
        match = re.search(r'SYSTEM(.*?)PARAMETER', text, flags=re.DOTALL)
        return match.group(1).strip() if match else None  # Entfernt unnötige Leerzeichen
    
    def finish_on_number(self, text):
        return text[-1].isdigit() if text else False
    
    def is_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

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
        retriever = vectorstore.as_retriever(k=self.given_results, similarity_threshold=0.7)

        # Define the prompt template for the LLM
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks.
            Use the following documents and conversation history to answer the question.
            Don't give unnecessary Informations.
            Answer always in the language of the question.
            Don't repeat the question!
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
        # Initialize the LLM
        llm = ChatOllama(
            model=self.model,
            temperature=self.temperature,
            num_ctx=self.num_ctx,
            format=self.format,
            verbose=self.verbose,
            seed=self.seed,
            num_predict=self.predict,
            top_k=self.top_k,
            top_p=self.top_p,
            repeat_penalty=self.repeat_penalty,
            repeat_last_n=self.repeat_last_n,
            num_gpu=self.num_gpu,
            stop=self.stop,
            keep_alive=self.keep_alive,
        )
        # Create a chain combining the prompt template and LLM
        rag_chain = prompt | llm | StrOutputParser()

        # Initialize the RAG application
        rag_application = self.RAGApplication(retriever, rag_chain)
        return rag_application

    # Method to perform a Google search
    def ddg_search(self):
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
        full_answer = ""
        self.conversation_history = self.conversation_history if self.history else []
        self.answer = rag_app.run(self.question, self.conversation_history)
        print(" " * 30, end="\r")
        for chunk in self.answer:
            print(chunk, end="", flush=True)
            full_answer += chunk
        print("\n")
        if self.history:
            self.conversation_history.append({"role": "user", "content": self.question})
            self.conversation_history.append({"role": "assistant", "content": full_answer})

    # Method to create a search query
    def search_query(self):
        to_date = date.today().strftime("%d.%m.%Y")
        prompt = f"""
    Todays date is {to_date}.
    Based on the following user question, formulate a concise and effective search query:
    "{self.question}"
    Your task:
    1. Create a search query of that will yield relevant results and contains all the relevant informations from the user question. If it is a topical question, it makes sense to include the date in the search query. If there is not a question or it is a personal question return **None**. If it does not make sense to make a web search to answer the question also return **None**
    2. Determine if a specific time range is needed for the search.
    Time range options:
    - 'd': Limit results to the past day. Use for very recent events or rapidly changing information.
    - 'w': Limit results to the past week. Use for recent events or topics with frequent updates.
    - 'm': Limit results to the past month. Use for relatively recent information or ongoing events.
    - 'y': Limit results to the past year. Use for annual events or information that changes yearly.
    - 'none': No time limit. Use for historical information or topics not tied to a specific time frame.
    """
        
        if self.history:
            message = self.conversation_history.copy()
            message.append({"role": "user", "content": prompt})

        # response = ollama.chat(
        #     messages=message,
        #     model=self.model,
        #     format=self.Query.model_json_schema(),
        #     options={"temperature": 0.5, "num_ctx": self.num_ctx}
        # )

        response = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Query.model_json_schema(), verbose=False, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=0.5, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(message if self.history else prompt)

        format_query = self.Query.model_validate_json(response.content)
        self.query = format_query.search_query
        self.time = format_query.timerange

        if self.debug:
            print(self.query)
            print(self.time)

    def commands(self):
        if self.question == "/help" or self.question == "/?":
            print('''Available Commands:
/set            Set session variables
/show           Show model information
/load <model>   Load a session or model
/clear          Clear session context
/bye            Exit
/?, /help       Help for a command

Use """ to begin a multi-line message.\n''')
            
        elif self.question == "/bye":
            sys.exit()
        elif self.question == "/clear":
            self.conversation_history = []
            print("Cleared session context")
        elif self.question.startswith("/load"):
            model = self.question.removeprefix("/load ").strip()
            if model:
                print(f"Loading model '{model}'")
                self.get_model(model)
                self.model = model   
            
        elif self.question.startswith("/show"):
            if self.question.rstrip().endswith("info"):
                show = ollama.show(model=self.model)
                print(f""" Model
architecture        {show.modelinfo["general.architecture"]}
parameters          {show.details.parameter_size}
context length      {show.modelinfo[f"{show.details.family}.context_length"]}
embedding length    {show.modelinfo[f"{show.details.family}.embedding_length"]}
quantization        {show.details.quantization_level}""")
                
            elif self.question.rstrip().endswith("license"):
                show = ollama.show(model=self.model)
                licence = show.license if show.license else "No license was specified for this model."
                print(licence)
                
            elif self.question.rstrip().endswith("modelfile"):
                show = ollama.show(model=self.model)
                print(show.modelfile)
                
            elif self.question.rstrip().endswith("parameters"):
                show = ollama.show(model=self.model)
                parameters = show.parameters if show.parameters else "No parameters were specified for this model."
                print(parameters)
                
            elif self.question.rstrip().endswith("system"):
                show = ollama.show(model=self.model)
                system = self.extract_between_system_and_parameter(show.modelfile) if show.modelfile else "No system message was specified for this model."
                print(system)
                print("")
                
            elif self.question.rstrip().endswith("template"):
                show = ollama.show(model=self.model)
                print(show.template)
                
            else:
                print("""Available Commands:
/show info         Show details for this model
/show license      Show model license
/show modelfile    Show Modelfile for this model
/show parameters   Show parameters for this model
/show system       Show system message
/show template     Show prompt template\n""")
                
        elif self.question.startswith("/set"):
            if self.question.rstrip().endswith("debug"):
                self.debug = True
            elif self.question.rstrip().endswith("nodebug"):
                self.debug = False
            elif self.question.rstrip().endswith("history"):
                self.history = True
            elif self.question.rstrip().endswith("nohistory"):
                self.history = False
            elif self.question.rstrip().endswith("websearch"):
                self.websearch_on = True
            elif self.question.rstrip().endswith("nowebsearch"):
                self.websearch_on = False
            elif self.question.rstrip().endswith("format json"):
                self.format = "json"
            elif self.question.rstrip().endswith("noformat"):
                self.format = None
            elif self.question.rstrip().endswith("verbose"):
                self.verbose = True
            elif self.question.rstrip().endswith("quiet"):
                self.verbose = False
            elif self.question.startswith("/set system"):
                system = self.question.removeprefix("/set system ").strip()
                if self.conversation_history[0]["role"] == "system":
                    self.conversation_history.pop(0)
                self.conversation_history.insert(0, {"role": "system", "content": system})
            elif self.question.startswith("/set parameter"):
                command = self.question.removeprefix("/set parameter ").strip()
                if command.startswith("seed"):
                    temp = command.removeprefix("seed ").strip()
                    temp = int(temp) if temp.isnumeric() else self.seed
                    self.seed = temp if temp == 0 else self.seed
                elif command.startswith("num_predict"):
                    temp = command.removeprefix("num_predict ").strip()
                    temp = int(temp) if temp.isnumeric() else self.predict
                    self.predict = temp if temp > 0 else self.predict
                elif command.startswith("top_k"):
                    temp = command.removeprefix("top_k ").strip()
                    temp = int(temp) if temp.isnumeric() else self.top_k
                    self.top_k = temp if temp > 0 else self.top_k
                elif command.startswith("top_p"):
                    temp = command.removeprefix("top_p ").strip()
                    temp = float(temp) if temp.replace('.', '', 1).isnumeric() else self.top_p
                    self.top_p = temp if temp > 0 else self.top_p
                # elif command.startswith("min_p"):
                #     temp = command.removeprefix("min_p ").strip()
                #     temp = float(temp) if temp.replace('.', '', 1).isnumeric() else self.min_p
                #     self.min_p = temp if temp > 0 else self.min_p
                elif command.startswith("num_ctx"):
                    temp = command.removeprefix("num_ctx ").strip()
                    temp = int(temp) if temp.isnumeric() else self.num_ctx
                    self.num_ctx = temp if temp > 0 else self.num_ctx
                elif command.startswith("temperature"):
                    temp = command.removeprefix("temperature ").strip()
                    temp = float(temp) if temp.replace('.', '', 1).isnumeric() else self.temperature
                    self.temperature = temp if temp > 0 else self.temperature
                elif command.startswith("repeat_penalty"):
                    temp = command.removeprefix("repeat_penalty ").strip()
                    temp = float(temp) if temp.replace('.', '', 1).isnumeric() else self.repeat_penalty
                    self.repeat_penalty = temp if temp > 0 else self.repeat_penalty
                elif command.startswith("repeat_last_n"):
                    temp = command.removeprefix("repeat_last_n ").strip()
                    temp = int(temp) if temp.isnumeric() else self.repeat_last_n
                    self.repeat_last_n = temp if temp > 0 else self.repeat_last_n
                elif command.startswith("num_gpu"):
                    temp = command.removeprefix("num_gpu ").strip()
                    temp = int(temp) if temp.isnumeric() else self.num_gpu
                    self.num_gpu = temp if temp > 0 else self.num_gpu
                elif command.startswith("stop"):
                    temp = command.removeprefix("stop ").strip()
                    temp = temp.split()
                    self.stop = temp if temp else self.stop
                elif command.startswith("num_results"):
                    temp = command.removeprefix("num_results ").strip()
                    temp = int(temp) if temp.isnumeric() else self.num_results
                    self.num_results = temp if temp > 0 else self.num_results
                elif command.startswith("given_results"):
                    temp = command.removeprefix("given_results ").strip()
                    temp = int(temp) if temp.isnumeric() else self.given_results
                    self.given_results = temp if temp > 0 else self.given_results
                else:
                    print("""Available Parameters:
/set parameter seed <int>             Random number seed
/set parameter num_predict <int>      Max number of tokens to predict
/set parameter top_k <int>            Pick from top k num of tokens
/set parameter top_p <float>          Pick token based on sum of probabilities
/set parameter num_ctx <int>          Set the context size
/set parameter temperature <float>    Set creativity level
/set parameter repeat_penalty <float> How strongly to penalize repetitions
/set parameter repeat_last_n <int>    Set how far back to look for repetitions
/set parameter num_gpu <int>          The number of layers to send to the GPU
/set parameter stop <string> <string> ...   Set the stop parameters
/set parameter num_results <int>      Number of websites that are fetched
/set parameter given_results <int>    Number of websites that the llm gets\n""")

            else:
                print("""Available Commands:
  /set parameter ...     Set a parameter
  /set system <string>   Set system message
  /set history           Enable history
  /set nohistory         Disable history
  /set format json       Enable JSON mode
  /set noformat          Disable formatting
  /set verbose           Show LLM stats
  /set quiet             Disable LLM stats\n""")
                
        else:
            print(f"Unknown command '{self.question}'. Type /? for help")