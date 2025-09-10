import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain_pinecone import PineconeVectorStore
from langchain import hub

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

load_dotenv()

def main():
    print("Hello from langchain-course!")


if __name__ == "__main__":
    print("Retrieving...")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatDeepSeek(model="deepseek-chat")

    query = "Please introduce me to the history of Normans in Sicily."
    chain = PromptTemplate.from_template(template=query) | llm
    #result = chain.invoke(input={})
    #print(result.content)

    vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    result = retrieval_chain.invoke(input={"input":query})

    print(result)
