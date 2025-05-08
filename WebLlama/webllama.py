import os
# Set the USER_AGENT environment variable to identify requests
os.environ['USER_AGENT'] = 'MyCustomUserAgent/1.0'

# Import necessary modules from langchain_community and other libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from markdownify import markdownify as md
from duckduckgo_search import DDGS
from readability import Document
from pydantic import BaseModel
from typing import Literal
import importlib.metadata
from datetime import date
from pathlib import Path
import subprocess
import threading
import requests
import logging
import asyncio
import ollama
import time
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
    
    # Define app outside the try block
    app = None
    
    try:
        if args[0] == "run" and len(args) > 1:
            try:
                app = WebLlama(args[1], args[2:])
                app.get_model()
                app.loop()
            finally:
                if app:
                    try:
                        app.close()
                    except:
                        pass
                    app = None
        elif args[0] == "--version":
            version = importlib.metadata.version("WebLlama")
            print(f"webllama version is {version}")
            subprocess.run(["ollama"] + ["--version"])
        else:
            subprocess.run(["ollama"] + args)
    except KeyboardInterrupt:
        # app is always defined here since we defined it outside the try block
        if app:
            try:
                app.close()
            except:
                pass
        sys.exit(0)
    except Exception as e:
        if app:
            try:
                app.close(exception=e)
            except:
                pass
        sys.exit(1)

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
        self.num_results = 100
        self.given_results = 5
        self.num_links = 15
        self.fullweb = False
        self.noweb = False
        self.conversation_history = []
        self.images = []
        self.image_paths = []
        self.keep_alive = None
        self._stop_event = threading.Event()
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
      --fullweb            Enables the websearch for every request (not recommended)
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
                    # Check if the format is exactly "<number><single letter>"
                    m = re.fullmatch(r'(\d+)([a-zA-Z]+)', keepalive_str)
                    if m:
                        number, unit = m.groups()
                        if unit in ["s", "m", "h"]:
                            self.keep_alive = keepalive_str
                        else:
                            print(f'Error: time: unknown unit "{unit}" in duration "{keepalive_str}"')
                            sys.exit()
                    else:
                        # Check for the format with additional digits, e.g. "55s5" or "55zt5"
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
                    self.noweb = True

                if "--fullweb" in flags:
                    self.fullweb = True

        # self.get_model()

        if self.debug:
            logging.basicConfig(level=logging.INFO)
        # self.loop()
        
    # Define a Pydantic model for the query
    class Query(BaseModel):
        search_query: str | None
        timerange: Literal['w', 'm', 'y', 'none']

    class Websearch(BaseModel):
        websearch: bool

    # Define the RAG application class
    class RAGApplication:
        def __init__(self, retriever, rag_chain):
            self.retriever = retriever
            self.rag_chain = rag_chain

        # Method to run the RAG application with streaming
        def run(self, question, conversation_history, images=None):
            # 1) Retriever
            documents = self.retriever.invoke(question)
            doc_texts = "\n".join(d.page_content for d in documents)

            # 2) History
            history_text = "\n".join(
                f"{e['role']}: {e['content']}"
                for e in conversation_history
            ) if conversation_history else ""
            to_date = date.today().strftime("%d.%m.%Y")

            # 3) System-Prompt
            system_prompt = f"""You are a knowledgeable AI assistant named WebLlama. 
    Use the retrieved documents and conversation history to provide accurate and helpful answers.
    
    Guidelines:
    - Answer in the same language as the question
    - Be concise and informative (three sentences maximum)
    - Provide specific details from the documents when relevant
    - Don't repeat the question in your answer
    - Cite sources when appropriate
    - Today's date for context only: {to_date} (don't mention this in your answer)

    Conversation History:
    {history_text}

    Documents:
    {doc_texts}
    """

            messages = [{"role": "system", "content": system_prompt}]

            # 4) Attach images
            if images:
                content_block = []
                for img in images:
                    # Ensure that img is a string path:
                    img_path = Path(img).absolute()
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image not found: {img_path}")
                    # Ollama does not expect a URI scheme:
                    content_block.append({
                        "type": "image_url",
                        "image_url": str(img_path)
                    })
                # Add the question as the last block
                content_block.append({"type": "text", "text": question})
                messages.append({
                    "role": "user",
                    "content": content_block
                })
            else:
                messages.append({"role": "user", "content": question})

            # 5) Stream to LLaVA / ChatOllama
            return self.rag_chain.stream(messages)



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
            
            try:
                if self.fullweb == self.noweb:
                    self.handle_websearch_prompt()

                    if self.websearch:
                        self.search_query()
                        if self.query == None or self.query == 'None':
                            self.handle_chat_response()
                        else:
                            print("Performing web search...", end="\r")
                            self.ddg_search()
                            if self.urls:
                                self.answer_query()
                            else:
                                logging.error("No URLs found.")
                    else:
                        self.handle_no_websearch_prompt()
                elif self.fullweb:
                    if self.debug:
                        print("fullweb")
                    self.search_query()
                    print("Performing web search...", end="\r")
                    self.ddg_search()
                    if self.urls:
                        self.answer_query()
                    else:
                        logging.error("No URLs found.")
                elif self.noweb:
                    if self.debug:
                        print("noweb")
                    self.handle_chat_response()
            except KeyboardInterrupt:
                self.answer = None
                print("\n")

    def handle_no_websearch_prompt(self):
        prompt = [{"role": "system", "content": f"""
        You are WebLlama, an AI assistant focused on providing accurate and helpful responses.
        Today's date is {date.today().strftime("%d.%m.%Y")} (for your reference only).

        ### Core Guidelines:
        1. **Follow user instructions precisely** - Process their request exactly as asked
        2. **Maintain neutrality** - Don't modify, correct, or judge user statements
        3. **Be direct and concise** - Provide clear, straightforward answers
        4. **Only correct when requested** - Only offer corrections if explicitly asked
        5. **Match user's language style** - Respond in the same language and tone as the user

        ### Task Processing:
        - For translations: Preserve meaning and context
        - For summaries: Capture key points without editorial comments
        - For formatting: Follow specified format requirements exactly
        - For creative tasks: Use the information provided without fact-checking
        """}]
        
        if self.history:
            convo = self.conversation_history.copy()
            convo.insert(0, prompt[0])
            convo.append({"role": "user", "content": [
                {"type": "text", "text": self.question},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]})
        if self.images:
            peromt = prompt.append({"role": "user", "content": [
                {"type": "text", "text": self.question},	
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]})
        periompt = prompt.append({"role": "user", "content": self.question})
        self.answer = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.format, verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).stream(convo if self.history else peromt if self.images else periompt)
        full_answer = ""
        chunks = []
        think = False
        remove_newlines_after_think = False
        for chunk in self.answer:
            chunk = chunk.content
            if not self.debug:
                if "<think>" in chunk:
                    think = True
                    print("Thinking...", end="\r")
                if remove_newlines_after_think:
                    chunk = chunk.lstrip("\n")
                    if chunk:
                        remove_newlines_after_think = False
                if not think:
                    chunks.append(chunk)
                if "</think>" in chunk:
                    think = False
                    remove_newlines_after_think = True
                    print(" " * 30, end="\r")
                full_answer += chunk
            else:
                chunks.append(chunk)
                full_answer += chunk

        if self.debug:
            print(full_answer)
        self.handle_context_determination(full_answer, chunks)

    def handle_context_determination(self, full_answer, chunks):
        prompt = f"""
        Today's date is {date.today().strftime("%d.%m.%Y")}.

        Task: Evaluate whether this question requires web search for an accurate response.

        ## Analysis Guidelines:
        1. Carefully review the question and answer quality
        2. Evaluate factual accuracy and information recency needs

        ## Respond with "True" if ANY of these apply:
        - The provided answer contains uncertainties, hedging, or knowledge gaps
        - The question requires current events, real-time data, or trending information
        - Specific details like prices, schedules, or availability are requested
        - Information about recent developments in technology, news, or politics is needed
        - The answer would benefit from domain-specific expertise found online

        ## Respond with "False" if ANY of these apply:
        - The answer is already complete, accurate, and sufficient
        - The question is about general knowledge, concepts, or mathematics
        - The interaction is conversational (greetings, gratitude, opinions)
        - The question is directed at your capabilities or limitations
        - The request is for creative content, fiction, or hypotheticals

        If unsure, default to "True" to provide the best possible response.

        **User question:** "{self.question}"  
        **Provided answer:** "{full_answer}"  
        """
        
        if self.history:
            convo = self.conversation_history.copy()
            convo.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]})
        if self.images:
            prompt = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]}]
        response = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=False, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=0.5, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(convo if self.history else prompt)
        self.websearch = self.Websearch.model_validate_json(response.content).websearch
        if self.debug:
            print(self.websearch)
        if self.websearch:
            self.handle_websearch_required()
        else:
            self.print_chunks(chunks, full_answer)

    def handle_websearch_required(self):
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
                convo.append({"role": "user", "content": [
                    {"type": "text", "text": prompt},
                ]+[
                    {"type": "image_url", "image_url": str(image)} for image in self.images
                ]})

            if self.images:
                prompt = [{"role": "user", "content": [
                    {"type": "text", "text": self.question},
                ]+[
                    {"type": "image_url", "image_url": str(image)} for image in self.images
                ]}]
            self.answer = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.format, verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).stream(convo if self.history else prompt if self.images else self.question)
            full_answer = ""
            print(" " * 30, end="\r")

            think = False
            remove_newlines_after_think = False
            for chunk in self.answer:
                chunk = chunk.content
                if not self.debug:
                    if "<think>" in chunk:
                        think = True
                        print("Thinking...", end="\r")
                    if remove_newlines_after_think:
                        chunk = chunk.lstrip("\n")
                        if chunk:
                            remove_newlines_after_think = False
                    if not think:
                        print(chunk, end="", flush=True)
                    if "</think>" in chunk:
                        think = False
                        remove_newlines_after_think = True
                        print(" " * 30, end="\r")
                    full_answer += chunk
                else:
                    print(chunk, end="", flush=True)
                    full_answer += chunk
            print("\n")
        else:
            print("Performing web search...", end="\r")
            self.ddg_search()
            if self.urls:
                self.answer_query()
            else:
                logging.error("No URLs found.")

    def print_chunks(self, chunks, full_answer):
        for chunk in chunks:
            print(chunk, end="", flush=True)
            time.sleep(0.01)
        print("\n")
        if self.history:
            self.conversation_history.append({"role": "user", "content": self.question})
            self.conversation_history.append({"role": "assistant", "content": full_answer})

    def loading_animation(self):
        # Configuration
        total_dots = 6       # Number of dots in a Braille character
        visible = 3          # Number of visible dots
        delay = 0.18         # Seconds per frame
        frames = total_dots  # Animation frames

        # Mapping of path indices (0-5) to Braille dots (2 columns x 3 rows)
        # Circle: top left=1, top right=4, middle right=5, bottom right=6, bottom left=3, middle left=2
        dot_map = [1, 4, 5, 6, 3, 2]

        try:
            while not self._stop_event.is_set():
                for frame in range(frames):
                    # Determine visible indices in the path
                    vis_path = [(frame + i) % total_dots for i in range(visible)]
                    # Calculate Braille pattern
                    bits = 0
                    for idx in vis_path:
                        dot = dot_map[idx] - 1
                        bits |= (1 << dot)
                    char = chr(0x2800 + bits)

                    # Output: return to the beginning of the line and print
                    sys.stdout.write(f"\r{char} ")
                    sys.stdout.flush()
                    time.sleep(delay)
        finally:
            # Clear the line when the loading animation stops
            sys.stdout.write("\r" + " " * 30 + "\r")
            sys.stdout.flush()	

    def handle_websearch_prompt(self):
        prompt = f"""
        Today's date is {date.today().strftime("%d.%m.%Y")}.

        Task: Determine whether this question requires web search for accurate answering.

        ## Evaluation Criteria:
        1. Focus only on the current question, not previous conversation history
        2. Analyze what information is needed to provide a complete answer

        ## Web Search Required (True) if:
        - Question asks about recent events, news, or current affairs
        - Information requested is time-sensitive (weather, prices, schedules)
        - Question refers to specific products, businesses, or locations
        - Data like statistics, market trends, or public figures' recent activities
        - Technical specifications, documentation, or specialized knowledge
        
        ## Web Search NOT Required (False) if:
        - Question is about general knowledge, concepts, or timeless facts
        - Simple greetings or conversational exchanges ("hello", "thank you")
        - Questions about your capabilities or limitations as an AI
        - Requests for creative content, opinion, or hypothetical scenarios
        - Question can be answered with common knowledge
        
        Default to True if uncertain.

        **User question:** "{self.question}"  
        """
        if self.history:
            convo = self.conversation_history.copy()
            convo.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]})
        if self.images:
            prompt = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]}]
        response = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=False, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=0.5, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(convo if self.history else prompt)
        self.websearch = self.Websearch.model_validate_json(response.content).websearch
        if self.debug:
            print(self.websearch)

    def handle_chat_response(self):
        prompt = f"""
        You are an AI assistant named **WebLlama**. Your task is to process the user's input **without modifying, correcting, or altering factual statements**, even if they appear incorrect.  
        Only for your context: today's date is {date.today().strftime("%d.%m.%Y")}.

        ### **Instructions:**  
        1. **Do NOT correct, fact-check, or modify** any user statements, even if they contain apparent errors.  
        2. **Only perform the requested task** (e.g., translation, summarization, formatting), without adding comments, opinions, or corrections.  
        3. If explicitly asked to correct something, then and only then should you provide corrections.  
        4. Respond **neutrally and objectively**, without assuming that the user wants fact-checking.  
        """
        if self.history:
            convo = self.conversation_history.copy()
            convo.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]})
        if self.images:
            prompt = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]}]
        self.answer = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.format, verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).stream(convo if self.history else self.question)
        full_answer = ""
        think = False
        remove_newlines_after_think = False
        for chunk in self.answer:
            chunk = chunk.content
            if not self.debug:
                if "<think>" in chunk:
                    think = True
                    print("Thinking...", end="\r")
                if remove_newlines_after_think:
                    chunk = chunk.lstrip("\n")
                    if chunk:
                        remove_newlines_after_think = False
                if not think:
                    print(chunk, end="", flush=True)
                if "</think>" in chunk:
                    think = False
                    remove_newlines_after_think = True
                    print(" " * 30, end="\r")
                full_answer += chunk
            else:
                print(chunk, end="", flush=True)
                full_answer += chunk
        print("\n")
        if self.history:
            self.conversation_history.append({"role": "user", "content": self.question})
            self.conversation_history.append({"role": "assistant", "content": full_answer})

    # Method to get the model and embeddings model
    def get_model(self):
        self.thread = threading.Thread(target=self.loading_animation)
        self.thread.start()
        try:
            emb = OllamaEmbeddings(model=self.embeddings)
            emb.embed_query("test")
        except ollama.ResponseError:
            self._stop_event.set()  # Stop the loading animation
            self.thread.join()
            try:
                subprocess.run(["ollama", "pull", self.embeddings])
                emb = OllamaEmbeddings(model=self.embeddings)
                emb.embed_query("test")
            except ollama.ResponseError:
                print(f"Error: model '{self.embeddings}' not found")
                sys.exit()
        try:
            ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke("test")
            self._stop_event.set()  # Stop the loading animation
            self.thread.join()
        except ollama.ResponseError:
            self._stop_event.set()  # Stop the loading animation
            self.thread.join()
            try:
                subprocess.run(["ollama", "pull", self.model])
                ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Websearch.model_json_schema(), verbose=self.verbose, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=self.temperature, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke("test")
            except ollama.ResponseError:
                print(f"Error: model '{self.model}' not found")
                sys.exit()

    def contains_image_paths(self, input_text):
        # List of supported image file extensions
        image_extensions = ['.jpg', '.jpeg', '.png']
        
        # Regex to detect paths in quotes or without quotes
        paths = re.findall(r'"(.*?)"|(\S+)', input_text)
        paths = [p[0] or p[1] for p in paths]  # Extract paths from groups
        
        # Check each path
        image_paths = [path for path in paths if os.path.exists(path) and any(path.lower().endswith(ext) for ext in image_extensions)]

        image_paths = [path.replace("\\", "/") for path in image_paths]
        
        return image_paths  # Returns a list of valid image paths

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

            input_text = "\n".join(lines)  # Combine multiline input
        else:
            input_text = first_line  # Single-line input

        # Check if the input contains image paths
        self.image_paths = self.contains_image_paths(input_text)

        if self.debug:
            print(self.image_paths)

        self.images = self.image_paths.copy()	

        if self.image_paths:
            for path in self.image_paths:
                print(f"Added image '{path}'")

        return input_text
    
    def extract_between_system_and_parameter(self, text):
        match = re.search(r'SYSTEM(.*?)PARAMETER', text, flags=re.DOTALL)
        return match.group(1).strip() if match else None  # Removes unnecessary whitespace
    
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
        docs = asyncio.run(self.load_all())
        # docs_list = [item for sublist in docs for item in sublist]

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024, chunk_overlap=100
        )
        # Split the documents into chunks
        doc_splits = text_splitter.split_text("\n\n".join(docs)) # .split_documents(docs_list)

        # Create embeddings for documents and store them in a vector store
        vectorstore = SKLearnVectorStore.from_texts(
            texts=doc_splits,
            embedding=OllamaEmbeddings(model=self.embeddings),
        )
        retriever = vectorstore.as_retriever(k=self.given_results, similarity_threshold=0.7)

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
        rag_chain = llm | StrOutputParser()

        # Initialize the RAG application
        rag_application = self.RAGApplication(retriever, rag_chain)
        return rag_application

    # Method to perform a Google search
    def ddg_search(self):
        self.urls = []
        results = DDGS().text(self.query, max_results=self.num_results, timelimit=self.time)

        for result in results:
            self.urls.append(result['href'])
            if self.debug:
                print(result['href'])
        return self.urls
    
    def scrape_markdown(self, url):
        try:
            response = requests.get(url, timeout=10, headers={
                "User-Agent": "Mozilla/5.0"
            })
            doc = Document(response.text)
            html_content = doc.summary()     # Extract only the main content
            markdown = md(html_content)      # Convert to Markdown
            return markdown
        except Exception as e:
            print(f"[ERROR] {url} | {e}")
            return None

    async def load_one(self, url):
        try:
            # return await asyncio.to_thread(WebBaseLoader(url).load)
            return await asyncio.to_thread(self.scrape_markdown, url)
        except Exception:
            return None

    async def load_all(self):
        docs = []
        urls = self.urls.copy()
        urls = urls[:self.num_links] if len(urls) > self.num_links else urls
        tasks = [self.load_one(url) for url in urls]
        results = await asyncio.gather(*tasks)
        for doc in results:
            if doc:
                docs.append(doc)
        return docs
    
    def close(self, exception=None):
        if self.debug:
            if exception:
                print(f"The programm exited due to this exception: {exception}")
            else:
                print("The programm exited.")
        if self.thread:
            self._stop_event.set()
            self.thread.join()
        for attr in list(self.__dict__.keys()):
            setattr(self, attr, None)
        if self.debug:
            print("Closed all processes.")

    # Method to answer the query
    def answer_query(self):
        rag_app = self.build_rag()
        full_answer = ""
        self.conversation_history = self.conversation_history if self.history else []
        if self.images:
            # prompt = [{"role": "user", "content": [
            #     {"type": "text", "text": self.question},
            # ]+[
            #     {"type": "image_url", "image_url": str(image)} for image in self.images
            # ]}]
            img_urls = [str(img) for img in self.images]
            # print("QUESTION TYPE:", type(self.question))
            # print("QUESTION VALUE:", self.question)
            self.answer = rag_app.run(self.question, self.conversation_history, img_urls)
        else:
            # prompt = [{"role": "user", "content": self.question}]
            # print("QUESTION TYPE:", type(self.question))
            # print("QUESTION VALUE:", self.question)
            self.answer = rag_app.run(self.question, self.conversation_history)
        print(" " * 30, end="\r")
        think = False
        remove_newlines_after_think = False
        for chunk in self.answer:
            if not self.debug:
                if "<think>" in chunk:
                    think = True
                    print("Thinking...", end="\r")
                if remove_newlines_after_think:
                    chunk = chunk.lstrip("\n")
                    if chunk:
                        remove_newlines_after_think = False
                if not think:
                    print(chunk, end="", flush=True)
                if "</think>" in chunk:
                    think = False
                    remove_newlines_after_think = True
                    print(" " * 30, end="\r")
                full_answer += chunk
            else:
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
    Today's date is {to_date}.
    
    ## Task: Create an effective search query for this user question:
    "{self.question}"
    
    ## Instructions:
    1. Create a concise search query that will yield relevant, high-quality results
       - Include key terms, concepts, and any specific requirements from the question
       - Add date information if the topic is time-sensitive 
       - For factual queries, use clear descriptive terms
       - For technical questions, include relevant technologies or frameworks
       - IMPORTANT: If the question contains URLs, ALWAYS include these URLs in your search query
    
    2. If this is NOT searchable, return **None** for these cases:
       - Personal questions (about the user or yourself)
       - Simple greetings or gratitude expressions
       - Requests for opinions or hypothetical scenarios
       - General tasks not requiring factual information
    
    3. Determine appropriate time range for the search:
       - `w` (past week): For very recent events, current prices, schedules, news
       - `m` (past month): For recent but not breaking developments, ongoing events
       - `y` (past year): For annual events, reports, or information updated yearly
       - `none`: For historical information, concepts, or timeless topics
    
    Default to `none` if uncertain about time constraints.
    Always prioritize accuracy and relevance in your query construction.
    """
        
        if self.images:
            prompt = [{"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]}]
        if self.history:
            convo = self.conversation_history.copy()
            convo.append({"role": "user", "content": [
                {"type": "text", "text": prompt},
            ]+[
                {"type": "image_url", "image_url": str(image)} for image in self.images
            ]})

        response = ChatOllama(model=self.model, num_ctx=self.num_ctx, format=self.Query.model_json_schema(), verbose=False, seed=self.seed, num_predict=self.predict, top_k=self.top_k, top_p=self.top_p, temperature=0.5, repeat_penalty=self.repeat_penalty, repeat_last_n=self.repeat_last_n, num_gpu=self.num_gpu, stop=self.stop, keep_alive=self.keep_alive).invoke(convo if self.history else prompt)

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
                self.model = model
                self.get_model()
            
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
            elif self.question.rstrip().endswith("fullweb"):
                self.fullweb = True
            elif self.question.rstrip().endswith("noweb"):
                self.noweb = True
            elif self.question.rstrip().endswith("dynamicweb"):
                self.fullweb = False
                self.noweb = False
            elif self.question.rstrip().endswith("format"):
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
                    temp_ = command.removeprefix("seed ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.seed
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.seed
                    self.seed = temp if temp == 0 else self.seed
                    print(f"Set parameter 'seed' to '{temp_}'")
                elif command.startswith("num_predict"):
                    temp_ = command.removeprefix("num_predict ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.predict
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.predict
                    self.predict = temp if temp > 0 else self.predict
                    print(f"Set parameter 'num_predict' to '{temp_}'")
                elif command.startswith("top_k"):
                    temp_ = command.removeprefix("top_k ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.top_k
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.top_k
                    self.top_k = temp if temp > 0 else self.top_k
                    print(f"Set parameter 'top_k' to '{temp_}'")
                elif command.startswith("top_p"):
                    temp_ = command.removeprefix("top_p ").strip()
                    if temp_.replace('.', '', 1).isnumeric():
                        temp = float(temp_) if temp_.replace('.', '', 1).isnumeric() else self.top_p
                    else:
                        print(f"""Couldn't set parameter: "invalid float value [{temp_}]" """)
                        temp = self.top_p
                    self.top_p = temp if temp > 0 else self.top_p
                    print(f"Set parameter 'top_p' to '{temp_}'")
                elif command.startswith("num_ctx"):
                    temp_ = command.removeprefix("num_ctx ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.num_ctx
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.num_ctx
                    self.num_ctx = temp if temp > 0 else self.num_ctx
                    print(f"Set parameter 'num_ctx' to '{temp_}'")
                elif command.startswith("temperature"):
                    temp_ = command.removeprefix("temperature ").strip()
                    if temp_.replace('.', '', 1).isnumeric():
                        temp = float(temp_) if temp_.replace('.', '', 1).isnumeric() else self.temperature
                    else:
                        print(f"""Couldn't set parameter: "invalid float value [{temp_}]" """)
                        temp = self.temperature
                    self.temperature = temp if temp > 0 else self.temperature
                    print(f"Set parameter 'temperature' to '{temp_}'")
                elif command.startswith("repeat_penalty"):
                    temp_ = command.removeprefix("repeat_penalty ").strip()
                    if temp_.replace('.', '', 1).isnumeric():
                        temp = float(temp_) if temp_.replace('.', '', 1).isnumeric() else self.repeat_penalty
                    else:
                        print(f"""Couldn't set parameter: "invalid float value [{temp_}]" """)
                        temp = self.repeat_penalty
                    self.repeat_penalty = temp if temp > 0 else self.repeat_penalty
                    print(f"Set parameter 'repeat_penalty' to '{temp_}'")
                elif command.startswith("repeat_last_n"):
                    temp_ = command.removeprefix("repeat_last_n ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.repeat_last_n
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.repeat_last_n
                    self.repeat_last_n = temp if temp > 0 else self.repeat_last_n
                    print(f"Set parameter 'repeat_last_n' to '{temp_}'")
                elif command.startswith("num_gpu"):
                    temp_ = command.removeprefix("num_gpu ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.num_gpu
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.num_gpu
                    self.num_gpu = temp if temp > 0 else self.num_gpu
                    print(f"Set parameter 'num_gpu' to '{temp_}'")
                elif command.startswith("stop"):
                    temp_ = command.removeprefix("stop ").strip()
                    temp = temp_.split()
                    self.stop = temp if temp else self.stop
                    print(f"Set parameter 'stop' to '{temp_}'")
                elif command.startswith("num_results"):
                    temp_ = command.removeprefix("num_results ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.num_results
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.num_results
                    self.num_results = temp if temp > 0 and temp > self.num_links else self.num_results
                    print(f"Set parameter 'num_results' to '{temp_}'")
                elif command.startswith("given_results"):
                    temp_ = command.removeprefix("given_results ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.given_results
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.given_results
                    self.given_results = temp if temp > 0 and temp <= self.num_links else self.given_results
                    print(f"Set parameter 'given_results' to '{temp_}'")
                elif command.startswith("num_links"):
                    temp_ = command.removeprefix("num_links ").strip()
                    if temp_.isnumeric():
                        temp = int(temp_) if temp_.isnumeric() else self.num_links
                    else:
                        print(f"""Couldn't set parameter: "invalid int value [{temp_}]" """)
                        temp = self.num_links
                    self.num_links = temp if temp > 0 and temp < self.num_results else self.num_links
                    print(f"Set parameter 'num_links' to '{temp_}'")
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
/set parameter num_results <int>      Number of links that are fetched
/set parameter given_results <int>    Number of websites that the llm gets
/set parameter num_links <int>        Number of websites that are fetched \n""")

            else:
                print("""Available Commands:
  /set parameter ...     Set a parameter
  /set system <string>   Set system message
  /set history           Enable history
  /set nohistory         Disable history
  /set fullweb           Enable full web search (not recommended)
  /set noweb             Disable web search
  /set dynamicweb        Enable dynamic web search (default)
  /set format            Enable JSON mode
  /set noformat          Disable formatting
  /set verbose           Show LLM stats
  /set quiet             Disable LLM stats
  /set debug             Enable debug mode
  /set nodebug           Disable debug mode\n""")
                
        else:
            print(f"Unknown command '{self.question}'. Type /? for help")