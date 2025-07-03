from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
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
    docs: list[Document] = []
    for d in tqdm(docs_gen, "Scrape websites", len(links)):
        if len(d.page_content) < 1000:  # likely blocked by website
            continue
        # filter only the news contents
        sentences = d.page_content.split("\n")
        sentences = [s for s in sentences if len(s) > 100]
        news = "\n".join(sentences)
        docs.append(d.model_copy(update={"page_content": news}))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    vector_store = InMemoryVectorStore(embeddings)
    for d in tqdm(split_docs, "Create vector store", disable=silent):
        vector_store.add_documents([d])
    return vector_store


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
