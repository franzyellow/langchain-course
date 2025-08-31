from dotenv import load_dotenv
load_dotenv()

from langchain import hub # hub is a collection of existing prompts
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_deepseek import ChatDeepSeek
from langchain_tavily import TavilySearch
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# Importing the customized classes in the other py files
from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse

tools = [TavilySearch()]
# Try different models if DeepSeek doesn't work well with ReAct format
llm = ChatDeepSeek(model="deepseek-chat")
react_prompt = hub.pull("hwchase17/react")
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
react_prompt_with_format_instructions = PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS,
    input_variables=["input", "agent_scratchpad", "tool_names"]
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format_instructions)
# Setting up the agent runtime, verbose=True will print reasoning process
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
extract_output = RunnableLambda(lambda x: x["output"])
parse_output = RunnableLambda(lambda x: output_parser.parse(x))

# Setting up the pipeline, so results from executor (a dict) will be passed as the input to saturate the lambda calculus in extract_output,
# and so on.
chain = agent_executor | extract_output | parse_output  

def main():
    result = chain.invoke(
        input={
            "input": "Search for 5 best-rated Asian restaurants in Utrecht.",
            }
        )
    print(result)


if __name__ == "__main__":
    main()
