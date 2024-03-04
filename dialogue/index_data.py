import uuid
from typing import List
from typing import Optional

import chromadb
import langdetect
from chromadb.utils import embedding_functions
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.data_anonymizer import PresidioReversibleAnonymizer
from transformers import AutoTokenizer

EMBEDDING_MODEL_NAME = "thenlper/gte-small"

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents(
    chunk_size: int,
    knowledge_base: List[LangchainDocument],
    tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
) -> List[LangchainDocument]:
    """

    :param chunk_size: int: The maximum number of tokens per chunk.
    :param knowledge_base: List[LangchainDocument]: The list of documents to split.
    :param tokenizer_name: Optional[str]: The name of the tokenizer to use for splitting the documents. (Default value = EMBEDDING_MODEL_NAME)
    :param chunk_size: int:
    :param knowledge_base: List[LangchainDocument]:
    :param tokenizer_name: Optional[str]:  (Default value = EMBEDDING_MODEL_NAME)
    :returns: List[LangchainDocument]: The list of split documents.

    """

    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=chunk_size // 10,
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    # Split documents
    docs_processed = [
        split_doc for doc in knowledge_base
        for split_doc in text_splitter.split_documents([doc])
    ]
    # Remove duplicates
    unique_texts = set()
    return [
        doc for doc in docs_processed
        if not (doc.page_content in unique_texts
                or unique_texts.add(doc.page_content))
    ]


def anonymize_documents(text: str) -> str:
    """Anonymize the given text.

    :param text: The text to be anonymized.
    :type text: str
    :param text: str:
    :returns: The anonymized text.

    """

    language = langdetect.detect(text)
    nlp_config = {
        "nlp_engine_name":
        "spacy",
        "models": [
            {
                "lang_code": "en",
                "model_name": "en_core_web_md"
            },
            {
                "lang_code": "es",
                "model_name": "es_core_news_md"
            },
            {
                "lang_code": "fr",
                "model_name": "fr_core_news_md"
            },
        ],
    }
    anonymizer = PresidioReversibleAnonymizer(
        analyzed_fields=[
            "PERSON",
            "PHONE_NUMBER",
            "EMAIL_ADDRESS",
            "CREDIT_CARD",
            "LOCATION",
            "DATE_TIME",
        ],
        languages_config=nlp_config,
    )
    return anonymizer.anonymize(text, language=language)


def indexer(docs: List[LangchainDocument], collection_name: str):
    """Add documents to the collection.

    :param docs: The list of documents to be added to the collection.
    :type docs: List[LangchainDocument]
    :param collection_name: The name of the collection.
    :type collection_name: str
    :param docs: List[LangchainDocument]:
    :param collection_name: str:
    :param docs: List[LangchainDocument]:
    :param collection_name: str:

    """
    client = chromadb.PersistentClient()
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL_NAME)
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=sentence_transformer_ef)

    # Add documents to the collection
    collection.add(
        ids=[str(uuid.uuid1()) for _ in docs],
        metadatas=[doc.metadata for doc in docs],
        documents=[anonymize_documents(doc.page_content) for doc in docs],
    )
