import os, time, mysql.connector, re, uuid, tiktoken
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever

load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY").strip()
print(f"os.environ['OPENAI_API_KEY']: {os.environ['OPENAI_API_KEY']}")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=None)

def chk_sid_regex():
    if re.search(r'[A-Za-z]{2}\d{4,}', st.session_state["dsmcb_sid"]):
        st.session_state["dsmcb_authenticated"] = True
        st.session_state["TM_Staff_ID"] = st.session_state["dsmcb_sid"]
    else:
        st.session_state["dsmcb_authenticated"] = False
        st.error("Not a valid TM Staff ID")

### ENTER TM STAFF ID
def enter_staff_id():
    if "dsmcb_authenticated" not in st.session_state:
        sid = st.text_input(label="Enter your TM Staff ID", value="", key="dsmcb_sid", on_change=chk_sid_regex)
        return False
    else:
        if st.session_state["dsmcb_authenticated"]:
            return True
        else:
            sid = st.text_input(label="Enter your TM Staff ID", value="", key="dsmcb_sid", on_change=chk_sid_regex)
            return False

### TO STREAM THE OUTPUT CHARACTERS
def stream_output(response):
    response = str(response).split(" ")
    for word in response:
        yield word + " "
        time.sleep(0.01)


### HISTORY AWARE RETRIEVER FUNCTION
def hist_aware_ret(prompt, conv_id):
    
    if 'vectordb' not in st.session_state:    
        st.session_state['vectordb'] = Chroma(persist_directory='./vectordbfiles_data/vectordbfiles_data', embedding_function=OpenAIEmbeddings())
    st.session_state['vectordb'].get()
    
    ### Contextualize question ###
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
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state['vectordb'].as_retriever(), contextualize_q_prompt
    )
    
    ### Answer question ###
    system_prompt = (
        "You are a fun and lively assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you don't know."
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
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    ### Statefully manage chat history ###
    if 'store' not in st.session_state:
        st.session_state['store'] = {}
    
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state['store']:
            st.session_state['store'][session_id] = ChatMessageHistory()
        return st.session_state['store'][session_id]
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    result = conversational_rag_chain.invoke(
        {"input": prompt},
        config={
            "configurable": {"session_id": conv_id}
        },
    )

    return result["answer"]

### MAIN FUNCTION
def run_chatbot():
    st.markdown("<h1 style='text-align: center; color: GREEN;'>DSM Centralised Chatbot</h3>", unsafe_allow_html=True)
    st.info("\tSend anything to begin the chat! For any issues, please [report here](https://forms.office.com/r/qjjDU0RLyS). For job inventory related questions, it is recommended to use Data Repository Detail app.", icon="ℹ️")

    if enter_staff_id():

        global dsmcb_sid
        dsmcb_sid = st.session_state["TM_Staff_ID"]
        
        if "conv_id" not in st.session_state:
            st.session_state["conv_id"] = str(uuid.uuid4())

        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        if prompt := st.chat_input("Ask anything!..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role":"user", "content":prompt})
            start_time = time.time()
    
            ### CALL FUNCTION TO THE HISTORY AWARE RETRIEVER ###
            answer = hist_aware_ret(prompt, st.session_state["conv_id"])
            time_taken = time.time() - start_time

            encoder = tiktoken.encoding_for_model('gpt-4')
            user_tokens = len(encoder.encode(prompt))
            response_tokens = len(encoder.encode(answer))
            total_tokens = user_tokens + response_tokens

            user_price_usd = (user_tokens/1000000) * 0.15
            response_price_usd = (response_tokens/1000000) * 0.6
            total_price_usd = user_price_usd + response_price_usd

            with st.chat_message("assistant"):
                st.write_stream(stream_output(answer))
    
            st.session_state.messages.append({"role":"assistant", "content":answer})

run_chatbot()
