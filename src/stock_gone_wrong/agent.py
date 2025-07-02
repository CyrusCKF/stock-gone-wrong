from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from tqdm import tqdm


def load_links(
    links: list[str],
    embeddings: Embeddings,
    chunk_size=1000,
    chunk_overlap=200,
    silent=False,
):
    """Download contents from Internet and store them in vector db"""
    loader = WebBaseLoader(links)
    docs_gen = loader.lazy_load()
    docs = []
    for d in tqdm(docs_gen, "Scrape websites", len(links), disable=silent):
        docs.append(d)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    vectorstore = InMemoryVectorStore(embeddings)
    for d in tqdm(split_docs, "Create vector DB", disable=silent):
        vectorstore.add_documents([d])
    return vectorstore


def model_qa(model: BaseChatModel, query: str, vectorstore: VectorStore):
    """Return (streaming_response, source)"""
    prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"""
    template = ChatPromptTemplate.from_messages([("system", prompt)])
    similar_docs = vectorstore.similarity_search(query, k=4)
    message = template.invoke(
        {
            "question": query,
            "context": "\n\n".join(doc.page_content for doc in similar_docs),
        }
    )
    response = model.stream(message)
    return response, similar_docs
