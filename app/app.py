"""
Streamlit app

Run this as follows:
> PYTHONPATH=. streamlit run app/app.py
"""

import os
import sys

# current_script = os.path.abspath(__file__)
# project_root = os.path.dirname(current_script)
# sys.path.append(project_root)
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from agent.config import set_environment
set_environment()

import streamlit as st
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from agent.agent import load_agent
from agent.utils import MEMORY

# Initialize session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="LangChain Question Answering", page_icon=":robot:")
st.header("Ask a research question!")

strategy = st.radio(
    "Reasoning strategy",
    ("plan-and-solve", "zero-shot-react"),
)

model_name = st.selectbox(
    "Select Model",
    [
        "gpt-3.5-turbo",
        "gpt-4",
        "Qwen/Qwen3-8B",
        "Qwen/Qwen2.5-7B",
    ],
    index=0
)

tool_names = st.multiselect(
    "Which tools do you want to use?",
    [
        "critical_search",
        "llm-math",
        "python_repl",
        "wikipedia",
        "arxiv",
        "google-search",
        "wolfram-alpha",
        "ddg-search",
        "openweathermap",
    ],
    ["ddg-search", "wikipedia", "arxiv", "openweathermap"],
)

if st.sidebar.button("Clear message history"):
    MEMORY.chat_memory.clear()

avatars = {"human": "user", "ai": "assistant"}
for msg in MEMORY.chat_memory.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

assert strategy is not None
agent_chain = load_agent(tool_names=tool_names, strategy=strategy, model_name=model_name)

if prompt := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            # Invoke the agent with the provided input and callback handler
            response = agent_chain.invoke(
                {"input": prompt}, 
                {"callbacks": [st_callback], "timeout": 120}
            )
            
            # Debugging: Print the raw response for inspection
            print(f"Raw response: {response}")

            # Process the response from the agent
            if isinstance(response, dict):
                if "action" in response:  # Check for plan-and-solve format
                    output = response.get("action_input", "未能生成答案，请尝试其他问题或工具。")
                else:
                    # Handle zero-shot-react or other strategies
                    output = response.get("output", response.get("answer", "未能生成答案，请尝试其他问题或工具。"))
            else:
                output = str(response)
            
            # Debugging: Print the processed output
            print(f"Processed output: {output}")
            
            # Append the assistant's response to the chat history
            st.session_state.chat_history.append({"role": "Assistant", "content": output})
            
            # Display the output
            st.write(output)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please try rephrasing your question, select different tools, or try a different model.")