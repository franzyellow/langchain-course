import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain import hub

load_dotenv()

if __name__ == "__main__":
    pdf_path = r"D:\work_study\Projects\langchain-course\China_Ethnic_Policy.pdf"
    print(f"Loading PDF from: {pdf_path}")
    
    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator='\n')
    docs = text_splitter.split_documents(documents=documents)
    print(f"Split into {len(docs)} chunks")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=os.environ.get("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-4o-2024-11-20")
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_china_ethnicity")

    new_vectorstore = FAISS.load_local("faiss_index_china_ethnicity", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain)

    res = retrieval_chain.invoke({"input": "中共的民族区域的设置和清朝与民国的政区制度有继承关系吗？"})
    print(res['answer'])



