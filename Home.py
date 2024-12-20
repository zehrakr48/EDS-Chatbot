import streamlit as st
from openai import OpenAI
import time
from typing_extensions import override
from openai import AssistantEventHandler
import io

# streamlit page settings
st.set_page_config(page_title="EDS Global Chatbot", page_icon="💬", layout="wide")

# initialize session state for chat history (if not already)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# openai client initialization
@st.cache_resource
def initialize_openai_client():
    return OpenAI(
        api_key=st.secrets.get(
            "OPENAI_API_KEY", ""
        )  # using streamlit secrets for the api key
    )


# assistant and vector store setup
@st.cache_resource
def setup_assistant(_client):
    # create assistant
    assistant = _client.beta.assistants.create(
        name="MyFileAssistant",
        instructions="You are a helpful assistant that uses uploaded files to answer questions about Vector DB.",
        model="gpt-4o-mini",
        tools=[{"type": "file_search"}],
    )

    # create vector store
    vector_store = _client.beta.vector_stores.create(name="EDSglobal")

    # upload files
    file_paths = ["C:/Users/zehrakr48/Desktop/files/tfile1.pdf"]
    file_streams = [open(file, "rb") for file in file_paths]
    file_batch = _client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    for stream in file_streams:
        stream.close()

    # update assistant with vector store
    assistant = _client.beta.assistants.update(
        assistant_id=assistant.id,
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}},
    )

    return assistant, vector_store


# custom event handler for capturing streaming response
class StreamlitEventHandler(AssistantEventHandler):
    def __init__(self):
        super().__init__()
        self.full_response = ""
        self.response_placeholder = st.empty()

    @override
    def on_text_delta(self, delta, snapshot):
        if delta.value:
            self.full_response += delta.value
            self.response_placeholder.markdown(self.full_response)


# main streamlit app
def main():
    st.title("📄 EDS Global Chatbot")

    # initialize openai client and assistant
    client = initialize_openai_client()
    assistant, vector_store = setup_assistant(client)

    # create thread
    thread = client.beta.threads.create()

    # sidebar for additional information
    st.sidebar.header("Chatbot Bilgisi")
    st.sidebar.info("Yüklenen PDF dosyası hakkında sorular sorun.")

    # display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # chat input
    if prompt := st.chat_input("Chatbot'a bir ileti gönder"):
        # add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # prepare assistant response
        with st.chat_message("assistant"):
            # create message in thread
            client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=prompt
            )

            # stream the response
            event_handler = StreamlitEventHandler()
            with client.beta.threads.runs.stream(
                thread_id=thread.id,
                assistant_id=assistant.id,
                instructions="You are an AI assistant helping users understand document contents.",
                event_handler=event_handler,
            ) as stream:
                stream.until_done()

            # add assistant response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": event_handler.full_response}
            )

    # cleanup button
    if st.sidebar.button("Sohbeti Sıfırla"):
        st.session_state.chat_history = []
        st.rerun()


# run the app
if __name__ == "__main__":
    main()
