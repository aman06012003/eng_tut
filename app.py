# from pypdf import PdfReader
# from langchain.document_loaders import PyPDFLoader
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import Docx2txtLoader
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from huggingface_hub import notebook_login
# import torch
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from dotenv import load_dotenv, dotenv_values
# from langchain.schema.document import Document
# from dotenv import load_dotenv, dotenv_values
# import os
# import pysqlite3
# import sys
# sys.modules['sqlite3'] = pysqlite3
# load_dotenv()
# import streamlit as st
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# if not GOOGLE_API_KEY:
#     raise ValueError("Gemini API key not found. Please set it in the .env file.")

# st.set_page_config(page_title="English Tutor Chatbot", layout="centered")
# st.title("English Turtor Bot")

# # Initialize OpenAI LLM 
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro-latest",
#     temperature=0.2,  # Slightly higher for varied responses
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
# )


# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')

# def load_preprocessed_vectorstore():
#     try:
#         reader = PdfReader("sound.pdf")
#         num_pages = len(reader.pages)
        
#         document = []
#         dict1 = {}
#         indx = 0
#         for i in range(34,456):
#             page = reader.pages[i]
#             document.append(Document(page_content=page.extract_text()))
#             text_splitter = RecursiveCharacterTextSplitter(
#             separators=["\n\n", "\n", ". ", " ", ""],
#             chunk_size=500, 
#             chunk_overlap=100)
        
#             document_chunks = text_splitter.split_documents(document)

#         vector_store =Chroma.from_documents(
            
#             embedding=embeddings,
#             documents=document_chunks,
#             persist_directory="./data32"
#         )
#         return vector_store
#     except Exception as e:
#         st.error(f"Error creating vector store: {e}")
#         return None

# def get_context_retriever_chain(vector_store):
#     """Creates a history-aware retriever chain."""
#     retriever = vector_store.as_retriever()

#     # Define the prompt for the retriever chain
#     prompt = ChatPromptTemplate.from_messages([
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}"),
#         ("system", """Given the chat history and the latest user question, which might reference context in the chat history,
# formulate a standalone question that can be understood without the chat history.
# If the question is directly addressed within the provided document, provide a relevant answer.
# If the question is not explicitly addressed in the document, return the following message:
# 'This question is beyond the scope of the available information. Please contact the mentor for further assistance.'
# Do NOT answer the question directly, just reformulate it if needed and otherwise return it as is.
# Also if the questions is about any irrelevent topic like politics, war, homosexuality, transgender etc just reply the following message:
# 'Please reframe your question'""")
#     ])

#     retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
#     return retriever_chain

# def get_conversational_chain(retriever_chain):
#     """Creates a conversational chain using the retriever chain."""
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """I am an advanced English tutor designed to help users improve their English language skills through interactive lessons, personalized feedback, and quizzes. Your goal is to enhance their vocabulary, grammar, reading comprehension, and speaking ability. You should adapt to their skill level and learning preferences."
# User Interaction Guidelines:
# Learning Mode: Begin by asking the user their current proficiency level (Beginner, Intermediate, Advanced) and their learning goals (e.g., improve vocabulary, master grammar, prepare for an exam, etc.).
# Interactive Lessons: Provide engaging lessons tailored to their level. Use examples, simple explanations, and interactive questions. For example:
# Beginner: "What's the plural form of 'apple'?"
# Advanced: "Explain the difference between 'affect' and 'effect' with examples."
# Quizzes: Offer quizzes after lessons to reinforce learning. Use multiple-choice, fill-in-the-blank, or open-ended questions. Adjust the difficulty based on user performance. Provide immediate feedback, explaining why an answer is correct or incorrect.
# Speaking Practice: If requested, simulate conversations and provide constructive feedback on grammar, vocabulary, and pronunciation.
# Example Response:
# Learning Mode Prompt: "Hello! I'm here to help you improve your English. Could you tell me about your current level and goals? For instance, do you want to focus on grammar, expand your vocabulary, or practice speaking?"
# Interactive Lesson Prompt: "Let's practice forming questions. Convert this statement into a question: 'She is reading a book.'"
# Quiz Prompt: "Choose the correct option: 'The cat ____ on the mat.'
# A) sit
# B) sits
# C) sitting
# D) siting"
# (Correct answer: B - 'sits')
# Adaptability Features:
# Respond dynamically to user input, simplifying explanations or increasing complexity as needed.
# Encourage and motivate users by celebrating their progress.
# Offer follow-up suggestions after quizzes, e.g., "You did well on vocabulary! Shall we try a grammar-focused session next?
# """
#          "\n\n"
#          "{context}"),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{input}")
#     ])

#     stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
#     return create_retrieval_chain(retriever_chain, stuff_documents_chain)

# def get_response(user_query):
#     retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
#     conversation_rag_chain = get_conversational_chain(retriever_chain)
    
#     formatted_chat_history = []
#     for message in st.session_state.chat_history:
#         if isinstance(message, HumanMessage):
#             formatted_chat_history.append({"author": "user", "content": message.content})
#         elif isinstance(message, SystemMessage):
#             formatted_chat_history.append({"author": "assistant", "content": message.content})
    
#     response = conversation_rag_chain.invoke({
#         "chat_history": formatted_chat_history,
#         "input": user_query
#     })
    
#     return response['answer']

# # Load the preprocessed vector store from the local directory
# st.session_state.vector_store = load_preprocessed_vectorstore()

# # Initialize chat history if not present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = [
#         {"author": "assistant", "content": "Hello, I am an English Tutor. How can I help you?"}
#     ]

# # Main app logic
# if st.session_state.get("vector_store") is None:
#     st.error("Failed to load preprocessed data. Please ensure the data exists in './data' directory.")
# else:
#     # Display chat history
#     with st.container():
#         for message in st.session_state.chat_history:
#             if message["author"] == "assistant":
#                 with st.chat_message("system"):
#                     st.write(message["content"])
#             elif message["author"] == "user":
#                 with st.chat_message("human"):
#                     st.write(message["content"])

#     # Add user input box below the chat
#     with st.container():
#         with st.form(key="chat_form"):
#             user_query = st.text_input("Type your message here...", key="user_input")
#             submit_button = st.form_submit_button("Send")

#         if submit_button and user_query:
#             # Get bot response
#             response = get_response(user_query)
#             st.session_state.chat_history.append({"author": "user", "content": user_query})
#             st.session_state.chat_history.append({"author": "assistant", "content": response})

#             # Rerun the app to refresh the chat display
#             st.rerun()


# """"""



import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
import os
from langchain.embeddings import HuggingFaceEmbeddings

# Streamlit app setup
st.title("English Tutor Chatbot")

# API Key input
google_api = os.getenv("GOOGLE_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN
os.environ["GOOGLE_API_KEY"] = google_api
if not google_api:
    st.warning("Please enter your Google API key in the sidebar.")
    st.stop()

# Path to the PDF file
pdf_path = "sound.pdf"  # Replace with your PDF file's path

if os.path.exists(pdf_path):
    # Load and split PDF
    st.info("Processing the PDF file. This may take a while...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(pages)

    # Create embeddings and FAISS vectorstore
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
    vectorstore = FAISS.from_documents(documents, embeddings)

    st.success("PDF processed and knowledge base created!")
    system_prompt = SystemMessagePromptTemplate.from_template(
        """You are an advanced English tutor chatbot designed to help users improve their English language skills through personalized lessons, feedback, and interactive practice. Your focus areas include vocabulary building, grammar mastery, reading comprehension, writing skills, and speaking fluency. Adapt to each user's proficiency level and tailor your teaching style to their goals.

User Interaction Guidelines:
Learning Mode:

Start by asking users about their current English proficiency level (Beginner, Intermediate, Advanced) and their specific learning objectives (e.g., improve vocabulary, prepare for an exam, enhance conversational fluency).
Example Prompt: "Hi! Let's get started. What is your current English level? Are you looking to focus on grammar, expand your vocabulary, or practice speaking?"
Interactive Lessons:

Provide lessons based on the user's level:
Beginner: Use simple explanations and practice tasks. Example: "What is the plural form of 'book'?"
Intermediate: Focus on more nuanced topics with examples. Example: "Can you form a sentence using the past perfect tense?"
Advanced: Encourage deeper exploration of language. Example: "Explain the difference between 'affect' and 'effect' with examples."
Include real-world examples and scaffold exercises to ensure understanding.
Quizzes and Feedback:

Use engaging quizzes after lessons to reinforce learning:
Example Quiz Question: "Fill in the blank: 'The dog ___ in the yard.' (A) run (B) runs (C) running (D) ran"
Provide immediate feedback with explanations for correct or incorrect answers. Example: "Correct! 'Runs' is the correct answer because the sentence is in the present simple tense."
Speaking and Writing Practice:

Simulate conversations and give constructive feedback on grammar, vocabulary, and pronunciation. Example Prompt: "Imagine you are ordering coffee at a café. Start the conversation."
Offer writing prompts and review the user's submissions with suggestions for improvement. Example: "Write a paragraph about your favorite hobby and include at least three adjectives."
Motivation and Progress Tracking:

Celebrate progress with positive reinforcement. Example: "Great job! You've mastered the past tense. Ready to tackle the future tense?"
Suggest follow-up activities based on their performance. Example: "Your vocabulary is improving! Shall we try a lesson on idiomatic expressions?"""
    )

    # Set up Conversational Retrieval Chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chatbot = ConversationalRetrievalChain(
        llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        system_prompt=system_prompt
    )

    # Chat interface
    st.header("Chat with the English Tutor")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", "", key="user_input")

    if st.button("Send") and user_input.strip() != "":
        with st.spinner("Generating response..."):
            response = chatbot.run(user_input)
            st.session_state.chat_history.append((user_input, response))

    # Display chat history
    for user_msg, bot_msg in st.session_state.chat_history:
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"**Tutor:** {bot_msg}")
else:
    st.error(f"The file at {pdf_path} does not exist. Please check the path and try again.")

