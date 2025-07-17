from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from tqdm import tqdm


def load_links(links: list[str], chunk_size=1000, chunk_overlap=200, silent=False):
    """Download contents from Internet and split them to chunks"""
    loader = WebBaseLoader(links)
    docs_gen = loader.lazy_load()
    docs: list[Document] = []
    for d in tqdm(docs_gen, "Scrape websites", len(links), disable=silent):
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
    splitted_docs = splitter.split_documents(docs)
    return splitted_docs


def model_qa(model: BaseChatModel, query: str, retriever: BaseRetriever):
    """Return (streaming_response, source)"""
    prompt = """You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"""
    template = ChatPromptTemplate.from_messages([("system", prompt)])
    similar_docs = retriever.invoke(query)
    message = template.invoke(
        {
            "question": query,
            "context": "\n\n".join(doc.page_content for doc in similar_docs),
        }
    )
    response = model.invoke(message)
    return response, similar_docs
