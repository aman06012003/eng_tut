from pypdf import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from huggingface_hub import notebook_login
import torch
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv, dotenv_values
import os
load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

reader = PdfReader("sound.pdf")

num_pages = len(reader.pages)

from langchain.schema.document import Document
document = []
dict1 = {}
indx = 0
for i in range(34,456):
  page = reader.pages[i]
  document.append(Document(page_content=page.extract_text()))


document_splitter=CharacterTextSplitter(separator='\n', chunk_size=500, chunk_overlap=100)
document_chunks=document_splitter.split_documents(document)
# sentence-transformers/all-mpnet-base-v
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
vectordb=Chroma.from_documents(document_chunks,embedding=embeddings, persist_directory='./data')
vectordb.persist()

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

retriever = vectordb.as_retriever()

prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", """Given the chat history and the latest user question, which might reference context in the chat history,
formulate a standalone question that can be understood without the chat history.
If the question is directly addressed within the provided document, provide a relevant answer.
If the question is not explicitly addressed in the document, return the following message:
'This question is beyond the scope of the available information. Please contact the mentor for further assistance.'
Do NOT answer the question directly, just reformulate it if needed and otherwise return it as is.
Also if the questions is about any irrelevent topic like politics, war, homosexuality, transgender etc just reply the following message:
'Please reframe your question'""")
    ])
retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

prompt = ChatPromptTemplate.from_messages([
        ("system", """I am an advanced English tutor designed to help users improve their English language skills through interactive lessons, personalized feedback, and quizzes. Your goal is to enhance their vocabulary, grammar, reading comprehension, and speaking ability. You should adapt to their skill level and learning preferences."

User Interaction Guidelines:

Learning Mode: Begin by asking the user their current proficiency level (Beginner, Intermediate, Advanced) and their learning goals (e.g., improve vocabulary, master grammar, prepare for an exam, etc.).
Interactive Lessons: Provide engaging lessons tailored to their level. Use examples, simple explanations, and interactive questions. For example:
Beginner: "What's the plural form of 'apple'?"
Advanced: "Explain the difference between 'affect' and 'effect' with examples."
Quizzes: Offer quizzes after lessons to reinforce learning. Use multiple-choice, fill-in-the-blank, or open-ended questions. Adjust the difficulty based on user performance. Provide immediate feedback, explaining why an answer is correct or incorrect.
Speaking Practice: If requested, simulate conversations and provide constructive feedback on grammar, vocabulary, and pronunciation.
Example Response:

Learning Mode Prompt: "Hello! I'm here to help you improve your English. Could you tell me about your current level and goals? For instance, do you want to focus on grammar, expand your vocabulary, or practice speaking?"
Interactive Lesson Prompt: "Let's practice forming questions. Convert this statement into a question: 'She is reading a book.'"
Quiz Prompt: "Choose the correct option: 'The cat ____ on the mat.'
A) sit
B) sits
C) sitting
D) siting"
(Correct answer: B - 'sits')
Adaptability Features:

Respond dynamically to user input, simplifying explanations or increasing complexity as needed.
Encourage and motivate users by celebrating their progress.
Offer follow-up suggestions after quizzes, e.g., "You did well on vocabulary! Shall we try a grammar-focused session next?
"""
         "\n\n"
         "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

stuff_documents_chain = create_stuff_documents_chain(llm, prompt)







import os  
import streamlit as st  
import google.generativeai as genai  
# from langchain_openai import OpenAI /
from langchain_google_genai import GoogleGenerativeAIEmbeddings  
from langchain_google_genai import ChatGoogleGenerativeAI  
# from langchain_openai import OpenAIEmbeddings  
from langchain_community.document_loaders import Docx2txtLoader   
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain_community.vectorstores import Chroma  
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  
from langchain_core.messages import HumanMessage, SystemMessage  
from langchain.chains import create_history_aware_retriever, create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain 
from dotenv import load_dotenv  
from langchain.embeddings import HuggingFaceEmbeddings
import pysqlite3
import sys
sys.modules['sqlite3'] = pysqlite3

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN ")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN 
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError("Gemini API key not found. Please set it in the .env file.")

st.set_page_config(page_title="English Tutor Chatbot", layout="centered")
st.title("English Turtor Bot")

# Initialize OpenAI LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,  # Slightly higher for varied responses
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Initialize embeddings using OpenAI
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

def load_preprocessed_vectorstore():
    try:
        document = []
        dict1 = {}
        indx = 0
        for i in range(34,456):
            page = reader.pages[i]
            document.append(Document(page_content=page.extract_text()))
            text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=3000, 
            chunk_overlap=1000)
        
            document_chunks = text_splitter.split_documents(document)

            vector_store = Chroma.from_documents(
            
            embedding=embeddings,
            documents=document_chunks,
            persist_directory="./data32"
            )
            return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_context_retriever_chain(vector_store):
    """Creates a history-aware retriever chain."""
    retriever = vector_store.as_retriever()

    # Define the prompt for the retriever chain
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("system", """Given the chat history and the latest user question, which might reference context in the chat history,
formulate a standalone question that can be understood without the chat history.
If the question is directly addressed within the provided document, provide a relevant answer.
If the question is not explicitly addressed in the document, return the following message:
'This question is beyond the scope of the available information. Please contact the mentor for further assistance.'
Do NOT answer the question directly, just reformulate it if needed and otherwise return it as is.
Also if the questions is about any irrelevent topic like politics, war, homosexuality, transgender etc just reply the following message:
'Please reframe your question'""")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_chain(retriever_chain):
    """Creates a conversational chain using the retriever chain."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", """I am an advanced English tutor designed to help users improve their English language skills through interactive lessons, personalized feedback, and quizzes. Your goal is to enhance their vocabulary, grammar, reading comprehension, and speaking ability. You should adapt to their skill level and learning preferences."

User Interaction Guidelines:

Learning Mode: Begin by asking the user their current proficiency level (Beginner, Intermediate, Advanced) and their learning goals (e.g., improve vocabulary, master grammar, prepare for an exam, etc.).
Interactive Lessons: Provide engaging lessons tailored to their level. Use examples, simple explanations, and interactive questions. For example:
Beginner: "What's the plural form of 'apple'?"
Advanced: "Explain the difference between 'affect' and 'effect' with examples."
Quizzes: Offer quizzes after lessons to reinforce learning. Use multiple-choice, fill-in-the-blank, or open-ended questions. Adjust the difficulty based on user performance. Provide immediate feedback, explaining why an answer is correct or incorrect.
Speaking Practice: If requested, simulate conversations and provide constructive feedback on grammar, vocabulary, and pronunciation.
Example Response:

Learning Mode Prompt: "Hello! I'm here to help you improve your English. Could you tell me about your current level and goals? For instance, do you want to focus on grammar, expand your vocabulary, or practice speaking?"
Interactive Lesson Prompt: "Let's practice forming questions. Convert this statement into a question: 'She is reading a book.'"
Quiz Prompt: "Choose the correct option: 'The cat ____ on the mat.'
A) sit
B) sits
C) sitting
D) siting"
(Correct answer: B - 'sits')
Adaptability Features:

Respond dynamically to user input, simplifying explanations or increasing complexity as needed.
Encourage and motivate users by celebrating their progress.
Offer follow-up suggestions after quizzes, e.g., "You did well on vocabulary! Shall we try a grammar-focused session next?
"""
         "\n\n"
         "{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_chain(retriever_chain)
    
    formatted_chat_history = []
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            formatted_chat_history.append({"author": "user", "content": message.content})
        elif isinstance(message, SystemMessage):
            formatted_chat_history.append({"author": "assistant", "content": message.content})
    
    response = conversation_rag_chain.invoke({
        "chat_history": formatted_chat_history,
        "input": user_query
    })
    
    return response['answer']

# Load the preprocessed vector store from the local directory
st.session_state.vector_store = load_preprocessed_vectorstore()

# Initialize chat history if not present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"author": "assistant", "content": "Hello, I am an English Tutor. How can I help you?"}
    ]

# Main app logic
if st.session_state.get("vector_store") is None:
    st.error("Failed to load preprocessed data. Please ensure the data exists in './data' directory.")
else:
    # Display chat history
    with st.container():
        for message in st.session_state.chat_history:
            if message["author"] == "assistant":
                with st.chat_message("system"):
                    st.write(message["content"])
            elif message["author"] == "user":
                with st.chat_message("human"):
                    st.write(message["content"])

    # Add user input box below the chat
    with st.container():
        with st.form(key="chat_form", clear_on_submit=True):
            user_query = st.text_input("Type your message here...", key="user_input")
            submit_button = st.form_submit_button("Send")

        if submit_button and user_query:
            # Get bot response
            response = get_response(user_query)
            st.session_state.chat_history.append({"author": "user", "content": user_query})
            st.session_state.chat_history.append({"author": "assistant", "content": response})

            # Rerun the app to refresh the chat display
            st.rerun()


""""""