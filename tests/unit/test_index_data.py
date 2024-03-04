from unittest.mock import patch, ANY

from langchain.docstore.document import Document as LangchainDocument

from dialogue.index_data import indexer
from dialogue.index_data import split_documents


def test_split_documents():
    # Create a mock knowledge_base
    knowledge_base = [
        LangchainDocument(
            page_content="This is a test document. It has multiple sentences."
        ),
        LangchainDocument(
            page_content="This is another test document. It also has multiple sentences."
        ),
    ]

    # Call the split_documents function
    chunk_size = 10  # Adjust this value as needed
    result = split_documents(chunk_size, knowledge_base)

    # Assert that the function returns the expected output
    assert isinstance(result, list)
    assert all(isinstance(doc, LangchainDocument) for doc in result)
    assert all(len(doc.page_content.split()) <= chunk_size for doc in result)


def test_indexer():
    # Create a mock docs
    docs = [
        LangchainDocument(
            page_content="This is a test document. It has multiple sentences."
        ),
        LangchainDocument(
            page_content="This is another test document. It also has multiple sentences."
        ),
    ]

    # Call the indexer function
    collection_name = "test_collection"

    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = mock_client.return_value.get_or_create_collection.return_value
        indexer(docs, collection_name)

        # Assert that the function has added the documents to the collection
        mock_collection.add.assert_called_once_with(
            ids=ANY,
            metadatas=[doc.metadata for doc in docs],
            documents=[doc.page_content for doc in docs],
        )