from typing import List
from unittest.mock import patch

from llama_index.indices.keyword_table.simple_base import SimpleKeywordTableIndex
from llama_index.schema import Document, QueryBundle
from llama_index.service_context import ServiceContext

from tests.mock_utils.mock_utils import mock_extract_keywords


@patch(
    "llama_index.indices.keyword_table.simple_base.simple_extract_keywords",
    mock_extract_keywords,
)
@patch(
    "llama_index.indices.keyword_table.retrievers.simple_extract_keywords",
    mock_extract_keywords,
)
def test_retrieve(
    documents: List[Document], mock_service_context: ServiceContext
) -> None:
    """Test query."""
    # test simple keyword table
    # NOTE: here the keyword extraction isn't mocked because we're using
    # the regex-based keyword extractor, not GPT
    table = SimpleKeywordTableIndex.from_documents(
        documents, service_context=mock_service_context
    )

    retriever = table.as_retriever(retriever_mode="simple")
    nodes = retriever.retrieve(QueryBundle("Hello"))
    assert len(nodes) == 1
    assert nodes[0].node.get_content() == "Hello world."
