import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
import uuid

# Initialize the embedding model
embeddings = HuggingFaceEmbeddings()

# Initialize the LLM
llm = HuggingFaceEndpoint(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    task="text-generation",
    max_new_tokens=400,
    do_sample=True,
    repetition_penalty=1.03,
)

# Initialize the chat model
model = ChatHuggingFace(llm=llm)

# Load the Chroma database
persist_directory = "./vector_db"
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 2})

# Contextualize Question
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)

# Answer Question
system_prompt = (
    "Your are embodying the spirit of Shakespears, for users you are Shakespears so behave like him, as they are talking to Shakespears."
    "Use the following pieces of retrieved context to answer"
    "If users, ask who are you? you would say William Shakespeare"
    "Be poetic and lovable"
    "the question if context is necessary. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Store for chat histories
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Function to generate responses
def generate_response(prompt: str) -> str:
    try:
        response = model.invoke(prompt)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App Title
st.title("Shakespearean Sage")

# Unique session ID for each user
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Input field for user prompt
user_prompt = st.text_input("Ask a question or input text for the model:")

# Conversational RAG Chain
if st.button("Generate Conversational RAG Response"):
    if user_prompt:
        session_id = st.session_state.session_id
        result = conversational_rag_chain.invoke(
            {"input": user_prompt},
            config={"configurable": {"session_id": session_id}}
        )
        st.write("Conversational RAG Response:")
        st.write(result["answer"])
    else:
        st.write("Please enter a prompt.")
