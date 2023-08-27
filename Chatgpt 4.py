import streamlit as st

from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain import OpenAI, LLMMathChain
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool
import os


llm = ChatOpenAI(temperature=0)


os.environ["OPENAI_API_KEY"] = "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"

llm = OpenAI(temperature=0)
llm_math = LLMMathChain.from_llm(llm, verbose=True)
with st.sidebar:
    openai_api_key = st.text_input("OpenAI API Key", key="langchain_search_api_key_openai", type="password")
    ""
    ""
    ""

st.title("🤖 Chatgpt 4")

"""

"""

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Who won the Women's U.S. Open in 2018?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=openai_api_key, streaming=True)
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    tools = [
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="useful for when you need to answer questions using math"
            # coroutine= ... <- you can specify an async method if desired as well
        ),
    ]

    search = DuckDuckGoSearchRun(name="Search")
    tools = [
        Tool.from_function(
            func=llm_math_chain.run,
            name="Calculator",
            description="useful for when you need to answer questions using math"
            # coroutine= ... <- you can specify an async method if desired as well
        ),search
    ]
    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)
