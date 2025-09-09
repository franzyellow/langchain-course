from dotenv import load_dotenv
from langchain.agents import tool
from langchain import hub
from langchain.prompts import PromptTemplate
from langchain_core.tools import render_text_description
from langchain_deepseek import ChatDeepSeek
from langchain.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from typing import Union, List
from langchain_core.tools import Tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.format_scratchpad import format_log_to_str
from callback import AgentCallbackHandler

load_dotenv()

def main():
    print("Hello from langchain-course!")
    

@tool # Transform the function into a tool that can be used by an LLM agent
def get_text_length(text: str) -> int:
    """Get the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"') # Remove the quotes and newlines from the text
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    main()
    tools = [get_text_length]

    template = """
    Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    Thought: {agent_scratchpad} 
    """
    # agent_scratchpad makes it possible for the LLM to make use of history of intermediate steps

    prompt = PromptTemplate(template=template).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))
    # At "tools=render_text_description(tools)", LangChain will transform StructuredTool objects into a string description for LLM input

    model = ChatDeepSeek(model="deepseek-chat", 
                         temperature=0, 
                         model_kwargs={"stop": ["\nObservation:", "Observation:", "Observation"]},
                         callbacks=[AgentCallbackHandler()])
    intermediate_steps = []
    # format the agent_scratchpad to a string for LLM input
    agent = ({"input": lambda x: x["input"], "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"])} 
            | prompt 
            | model 
            | ReActSingleInputOutputParser())

    # The whole thing below is the rationale behind the built-in AgentExecutor
    agent_step = ""
    while not isinstance(agent_step, AgentFinish):

        # “该变量的类型可以是 AgentAction 或 AgentFinish 其中之一”
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the text length of the word 'HELLO' in characters?"
                                                               ,"agent_scratchpad": intermediate_steps})
        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            observation = tool_to_use.func(str(tool_input))
            print(f"Observation: {observation}")
            intermediate_steps.append((agent_step, str(observation)))

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
    
