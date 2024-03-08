from typing import Optional

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


def get_message(
    query: str,
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    history: Optional[str] = TEMPLATE,
):
    """Returns the response from the chatbot based on the given query.

    :param query: The query to be passed to the chatbot.
    :type query: str
    :param tokenizer_name: The name of the tokenizer model to be used. Defaults to
    :type tokenizer_name: Optional[str]
    :param tokenizer_name: The name of the tokenizer model to be used. Defaults to
        EMBEDDING_MODEL_NAME.
    :type tokenizer_name: Optional[str]
    :param history: The history template for generating the chat history. Defaults to TEMPLATE.
    :type history: Optional[str]
    :param query: str:
    :param tokenizer_name: Optional[str]:  (Default value = EMBEDDING_MODEL_NAME)
    :param history: Optional[str]:  (Default value = TEMPLATE)
    :returns: The response from the chatbot.
    :rtype: str

    """
    embedding = HuggingFaceEmbeddings(model_name=tokenizer_name)
    vectorstore = Chroma(persist_directory="chroma", embedding_function=embedding)
    llm = Ollama(model="gemma:7b")

    retriever = vectorstore.as_retriever()

    # This controls how the standalone question is generated.
    # Should take `chat_history` and `question` as input variables.

    prompt = PromptTemplate.from_template(history)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain.invoke(query)
