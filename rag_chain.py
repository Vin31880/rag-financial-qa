from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    print(f"✅ Loaded {len(pages)} pages from {path}")
    return pages


def split_document(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=True
    )
    chunks = splitter.split_documents(pages)
    print(f"✅ Split into {len(chunks)} chunks")
    return chunks


def create_vectore_store(chunks):
    embedding = OllamaEmbeddings(model="mistral")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    print("✅ Vector store created and persisted")
    return vector_store



def build_qa_chain(vector_store):
    llm = OllamaLLM(model="mistral", temperature=0.3)
    prompt_template = """You are a financial document analyst.
    Use the following context to answer the question accurately and concisely.
    If you don't know the answer from the context, say "I cannot find that information in the document."

    Context: {context}
    Question: {question}
    Answer:"""
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )
    return qa_chain

if __name__ == '__main__':
    pages = load_pdf("data/apple_report.pdf")
    chunks = split_document(pages)
    vector_store = create_vectore_store(chunks)
    qa_chain = build_qa_chain(vector_store)

    # Test questions
    questions = [
        "What is the main topic of this document?",
        "What are the key financial highlights?",
        "What risks are mentioned in the document?"
    ]

    print("\n── Q&A Results ──────────────────────────────")
    for q in questions:
        print(f"\n❓ {q}")
        result = qa_chain.invoke({"query": q})
        print(f"💡 {result['result']}")
