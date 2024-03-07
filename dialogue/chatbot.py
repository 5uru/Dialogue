from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough

EMBEDDING_MODEL_NAME = "thenlper/gte-small"
TEMPLATE = """Answer the question based only on the following context:
{context}

Question: {question}
"""


def get_message(query: str):
    """Returns the response from the chatbot based on the given query.

    :param query: The query string to be passed to the chatbot.
    :type query: str
    :param query: str:
    :param query: str:
    :returns: The response generated by the chatbot.
    :rtype: str

    """
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma(persist_directory="chroma",
                         embedding_function=embedding)
    llm = Ollama(model="gemma:7b")

    retriever = vectorstore.as_retriever()

    # This controls how the standalone question is generated.
    # Should take `chat_history` and `question` as input variables.

    prompt = PromptTemplate.from_template(TEMPLATE)
    chain = ({
        "context": retriever,
        "question": RunnablePassthrough()
    }
             | prompt
             | llm
             | StrOutputParser())
    return chain.invoke("where did harrison work?")
