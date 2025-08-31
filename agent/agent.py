from typing import Literal, Dict, Any
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables import Runnable
from langchain_experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
from langchain.prompts import PromptTemplate

from agent.tool_loader import load_tools
from agent.utils import MEMORY  # Import shared memory instance
from agent.config import set_environment

set_environment()

# Define the type for reasoning strategies
ReasoningStrategies = Literal["zero-shot-react", "plan-and-solve"]

def create_llm(model_name: str) -> BaseLanguageModel:
    """Create LLM based on model_name."""
    if model_name.startswith("gpt-"):
        return ChatOpenAI(model=model_name, temperature=0, streaming=True)
    elif model_name.startswith("Qwen/"):
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            streaming=True,
        )
    else:
        return ChatOpenAI(model='gpt-3.5-turbo', temperature=0, streaming=True)

def load_agent(tool_names: list[str], strategy: ReasoningStrategies = "zero-shot-react", model_name: str = "gpt-3.5-turbo") -> Runnable:
    """Load and configure an agent with specified tools and reasoning strategy."""
    llm = create_llm(model_name)
    tools = load_tools(tool_names=tool_names, llm=llm)
    
    if strategy == "plan-and-solve":
        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        
        class PlanAndExecuteWrapper(Runnable):
            def __init__(self, planner, executor):
                self.planner = planner
                self.executor = executor
                self.plan_and_execute = PlanAndExecute(
                    planner=planner, 
                    executor=executor, 
                    verbose=True
                )
                self.plan_and_execute.memory = MEMORY
            
            def invoke(self, input: Dict[str, Any], config=None) -> Dict[str, Any]:
                original_question = input.get("input", "")
                result = self.plan_and_execute.invoke({"input": original_question}, config)
                
                if "output" not in result or not result["output"]:
                    if "intermediate_steps" in result:
                        for step in reversed(result["intermediate_steps"]):
                            if isinstance(step, dict) and "response" in step:
                                result["output"] = step["response"]
                                break
                
                return {
                    "output": result.get("output", "未能生成答案，请尝试其他问题或工具。"),
                    "intermediate_steps": result.get("intermediate_steps", [])
                }
            
            async def ainvoke(self, input: Dict[str, Any], config=None):
                return self.invoke(input, config)
        
        return PlanAndExecuteWrapper(planner, executor)
    
    prompt = hub.pull("hwchase17/react")
    return AgentExecutor(
        agent=create_react_agent(llm=llm, tools=tools, prompt=prompt), 
        tools=tools,
        handle_parsing_errors=True,
        max_iterations=15,
        early_stopping_method="generate",
        verbose=True,
        memory=MEMORY  # Use shared memory for conversation context
    )