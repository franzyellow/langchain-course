from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import re


load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """
    The Greco-Bactrian Kingdom (Greek: Βασιλεία τῆς Βακτριανῆς, romanized: Basileía tês Baktrianês, lit. 'Kingdom of Bactria') was a Greek state of the Hellenistic period located in Central-South Asia. 
    The kingdom was founded by the Seleucid satrap Diodotus I Soter in about 256 BC, and continued to dominate Central Asia until its fall around 120 BC.
    At its peak the kingdom consisted of present-day Afghanistan, Tajikistan, Uzbekistan and Turkmenistan, and for a short time, small parts of Kazakhstan, Pakistan and Iran. 
    An extension further east, with military campaigns and settlements, may have reached the borders of the Qin State in China by about 230 BC.
    """

    summary_template = f"""
        Given the information {information} about a historical fact, I would like you to create:
        1. A short summary
        2. The two most interesting points about it

        DO NOT PRODUCE ANYTHING ELSE!
        """

    final_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a concise assistant for text summarization."),
        ("human", summary_template),
    ])

    hf_llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen3-4B-Instruct-2507",
        task="conversational",  # Qwen3-8B 支持对话
        max_new_tokens=1024,
        temperature=0.0,
    )


    llm = ChatHuggingFace(llm=hf_llm)

    # 链式调用
    chain = final_prompt_template | llm
    response = chain.invoke({"information": information})
    raw_response = response.content
    # 删除 <think>...</think> 部分（跨多行用 DOTALL）
    clean_response = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL).strip()
    
    print(clean_response)



if __name__ == "__main__":
    main()
