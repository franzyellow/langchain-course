from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama


load_dotenv()

def main():
    print("Hello from langchain-course!")
    information = """
    The Greco-Bactrian Kingdom (Greek: Βασιλεία τῆς Βακτριανῆς, romanized: Basileía tês Baktrianês, lit. 'Kingdom of Bactria') was a Greek state of the Hellenistic period[2][3][4] located in Central-South Asia. 
    The kingdom was founded by the Seleucid satrap Diodotus I Soter in about 256 BC, and continued to dominate Central Asia until its fall around 120 BC.[a] 
    At its peak the kingdom consisted of present-day Afghanistan, Tajikistan, Uzbekistan and Turkmenistan, and for a short time, small parts of Kazakhstan, Pakistan and Iran. 
    An extension further east, with military campaigns and settlements, may have reached the borders of the Qin State in China by about 230 BC.
    """

    summary_template = f"""
    Given the information {information} about a historical fact, I would like you to create:
    1. A short summary
    2. The two most interesting points about it

    Strictly stay to this format and do not produce anything else.
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    llm = ChatDeepSeek(model="deepseek-chat", temperature=0)
    #llm = ChatOllama(model="gemma3:1b", temperature=0)
    chain = summary_prompt_template | llm # Inputing from template to LLM, leading to a runnable object
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()

