from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface.llms import HuggingFacePipeline
import os
import re, json

os.environ['TRANSFORMERS_CACHE'] = '/scratch-local/yhuang/huggingface_cache'
# 确保目录存在
os.makedirs('/scratch-local/yhuang/huggingface_cache', exist_ok=True)

load_dotenv()

def main():
    print("Hello from langchain-course!")

    information = """
    The Seleucid Empire (/sɪˈljuːsɪd/ sih-LEW-sid[8]) was a Greek state[9][10] in West Asia during the Hellenistic period. It was founded in 312 BC by the Macedonian general Seleucus I Nicator, following the division of the Macedonian Empire founded by Alexander the Great,[11][12][13][14] and ruled by the Seleucid dynasty until its annexation by the Roman Republic under Pompey in 63 BC.

    After receiving the Mesopotamian regions of Babylonia and Assyria in 321 BC, Seleucus I began expanding his dominions to include the Near Eastern territories that encompass modern-day Iraq, Iran, Afghanistan, Syria, and Lebanon, all of which had been under Macedonian control after the fall of the former Achaemenid Empire. At the Seleucid Empire's height, it had consisted of territory that covered Anatolia, Persia, the Levant, Mesopotamia, and what are now modern Kuwait, Afghanistan, and parts of Turkmenistan.

    The Seleucid Empire was a major center of Hellenistic culture. Greek customs and language were privileged; the wide variety of local traditions had been generally tolerated, while an urban Greek elite had formed the dominant political class and was reinforced by steady immigration from Greece.[14][15][16] The empire's western territories were repeatedly contested with Ptolemaic Egypt—a rival Hellenistic state. To the east, conflict with the Indian ruler Chandragupta of the Maurya Empire in 305 BC led to the cession of vast territory west of the Indus and a political alliance.

    In the early second century BC, Antiochus III the Great attempted to project Seleucid power and authority into Hellenistic Greece, but his attempts were thwarted by the Roman Republic and its Greek allies. The Seleucids were forced to pay costly war reparations and had to relinquish territorial claims west of the Taurus Mountains in southern Anatolia, marking the gradual decline of their empire. Mithridates I of Parthia conquered much of the remaining eastern lands of the Seleucid Empire in the mid-second century BC, including Assyria and what had been Babylonia, while the independent Greco-Bactrian Kingdom continued to flourish in the northeast. The Seleucid kings were thereafter reduced to a rump state in Syria after a civil war, until their conquest by Tigranes the Great of Armenia in 83 BC, and ultimate overthrow by the Roman general Pompey in 63 BC.
    """

    translation_template = """
        Given the information {information}, I would like you to return your response in the following JSON format:
        {{"Translation": "[THE TRANSLATION INTO SIMPLIFIED CHINESE]",
        "Keywords": "[UP TO 5 KEYWORDS IN SIMPLIFIED CHINESE]"}}

        Please strictly stick to this format and do not produce anything else.
        """

    final_prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for text translation and keyword retrieval."),
        ("human", translation_template),
    ])


    hf_llm = HuggingFacePipeline.from_model_id(
        model_id = "Qwen/Qwen3-4B-Instruct-2507",
        task="text-generation",
        pipeline_kwargs = {"do_sample": True, "max_new_tokens": 1024, "temperature": 0.3}
    )


    llm = ChatHuggingFace(llm=hf_llm)

    # 链式调用
    chain = final_prompt_template | llm
    response = chain.invoke({"information": information})
    raw_response = response.content
    # 提取第一个 {...} 块
    m = re.search(r"\{[\s\S]*\}", raw_response)
    if m:
        try:
            clean_response = json.loads(m.group(0))
        except json.JSONDecodeError:
            clean_response = m.group(0).strip()  # 如果解码失败，就原样返回字符串
    else:
        clean_response = raw_response.strip()
    
    print(clean_response)



if __name__ == "__main__":
    main()