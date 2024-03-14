"""Knowledge Graph Index.

Build a KG by extracting triplets, and leveraging the KG during query-time.

"""

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from llama_index.constants import GRAPH_STORE_KEY
from llama_index.core.base_retriever import BaseRetriever
from llama_index.data_structs.data_structs import KG
from llama_index.graph_stores.simple import SimpleGraphStore
from llama_index.graph_stores.types import GraphStore
from llama_index.indices.base import BaseIndex
from llama_index.prompts import BasePromptTemplate
from llama_index.prompts.default_prompts import DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
from llama_index.schema import BaseNode, IndexNode, MetadataMode
from llama_index.service_context import ServiceContext
from llama_index.storage.docstore.types import RefDocInfo
from llama_index.storage.storage_context import StorageContext
from llama_index.utils import get_tqdm_iterable
import json
logger = logging.getLogger(__name__)
from pinecone import Pinecone

pc = Pinecone(api_key="c6c12059-8962-4480-bd78-ebc90d0dc043")
pincone_index = pc.Index(host="dexterkg")

class KnowledgeGraphIndex(BaseIndex[KG]):
    """Knowledge Graph Index.

    Build a KG by extracting triplets, and leveraging the KG during query-time.

    Args:
        kg_triple_extract_template (BasePromptTemplate): The prompt to use for
            extracting triplets.
        max_triplets_per_chunk (int): The maximum number of triplets to extract.
        service_context (Optional[ServiceContext]): The service context to use.
        storage_context (Optional[StorageContext]): The storage context to use.
        graph_store (Optional[GraphStore]): The graph store to use.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        include_embeddings (bool): Whether to include embeddings in the index.
            Defaults to False.
        max_object_length (int): The maximum length of the object in a triplet.
            Defaults to 128.
        kg_triplet_extract_fn (Optional[Callable]): The function to use for
            extracting triplets. Defaults to None.

    """

    index_struct_cls = KG

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        objects: Optional[Sequence[IndexNode]] = None,
        index_struct: Optional[KG] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        kg_triple_extract_template: Optional[BasePromptTemplate] = None,
        max_triplets_per_chunk: int = 10,
        include_embeddings: bool = False,
        show_progress: bool = False,
        max_object_length: int = 128,
        kg_triplet_extract_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize params."""
        # need to set parameters before building index in base class.
        self.include_embeddings = include_embeddings
        self.max_triplets_per_chunk = max_triplets_per_chunk
        self.kg_triple_extract_template = (
            kg_triple_extract_template or DEFAULT_KG_TRIPLET_EXTRACT_PROMPT
        )
        # NOTE: Partially format keyword extract template here.
        self.kg_triple_extract_template = (
            self.kg_triple_extract_template.partial_format(
                max_knowledge_triplets=self.max_triplets_per_chunk
            )
        )
        self._max_object_length = max_object_length
        self._kg_triplet_extract_fn = kg_triplet_extract_fn

        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            objects=objects,
            **kwargs,
        )

        # TODO: legacy conversion - remove in next release
        if (
            len(self.index_struct.table) > 0
            and isinstance(self.graph_store, SimpleGraphStore)
            and len(self.graph_store._data.graph_dict) == 0
        ):
            logger.warning("Upgrading previously saved KG index to new storage format.")
            self.graph_store._data.graph_dict = self.index_struct.rel_map

    @property
    def graph_store(self) -> GraphStore:
        return self._graph_store

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        from llama_index.indices.knowledge_graph.retrievers import (
            KGRetrieverMode,
            KGTableRetriever,
        )

        if len(self.index_struct.embedding_dict) > 0 and "retriever_mode" not in kwargs:
            kwargs["retriever_mode"] = KGRetrieverMode.EMBEDDING

        return KGTableRetriever(self, object_map=self._object_map, **kwargs)

    def _extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        if self._kg_triplet_extract_fn is not None:
            return self._kg_triplet_extract_fn(text)
        else:
            return self._llm_extract_triplets(text)

    def _llm_extract_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract keywords from text."""
        print("text", text)
        response = self._service_context.llm.predict(
            self.kg_triple_extract_template,
            text=text,
        )
        return self._parse_triplet_response(
            response, max_length=self._max_object_length
        )

    @staticmethod
    def _parse_triplet_response(
        response: str, max_length: int = 128
    ) -> List[Tuple[str, str, str]]:
        knowledge_strs = response.strip().split("\n")
        results = []
        for text in knowledge_strs:
            if "(" not in text or ")" not in text or text.index(")") < text.index("("):
                # skip empty lines and non-triplets
                continue
            triplet_part = text[text.index("(") + 1 : text.index(")")]
            tokens = triplet_part.split(",")
            if len(tokens) != 3:
                continue

            if any(len(s.encode("utf-8")) > max_length for s in tokens):
                # We count byte-length instead of len() for UTF-8 chars,
                # will skip if any of the tokens are too long.
                # This is normally due to a poorly formatted triplet
                # extraction, in more serious KG building cases
                # we'll need NLP models to better extract triplets.
                continue

            subj, pred, obj = map(str.strip, tokens)
            if not subj or not pred or not obj:
                # skip partial triplets
                continue
            results.append((subj, pred, obj))
        return results

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> KG:
        """Build the index from nodes."""
        # do simple concatenation
        index_struct = self.index_struct_cls()
        nodes_with_progress = get_tqdm_iterable(
            nodes, self._show_progress, "Processing nodes"
        )
        for n in nodes_with_progress:
            #triplets = self._extract_triplets(
            text =  n.get_content(metadata_mode=MetadataMode.LLM)
            print(text)
            json_start = text.find('{')

            # Extract the JSON string
            json_str = text[json_start:]
            print(json_str)
            # Load the JSON string into a dictionary
            data_dict = json.loads(json_str)
            print(data_dict)
            #
            triplets = []
            # Create nodes and relationships for items
            case_id = data_dict['case_id']
            ua=data_dict["userAnnotation"]
            items = ua["items"]
            for key, value in items.items():
                    triplets.append((case_id, "HAS_" + key.upper(), value))

            line_items = ua["line_items"]
            for line_item in line_items:
                    item_code = line_item["item_code"]
                    
                    triplets.append((case_id, "HAS_ITEM_CODE", item_code))

                    for line_item_key, line_item_value in line_item.items():
                        if line_item_key != "item_code":
                            triplets.append((item_code, "HAS_" + line_item_key.upper(), line_item_value))
            
            #logger.debug(f"> Extracted triplets: {triplets}")
            vectors = []
            for tr_i,tr in enumerate(triplets):
                triplet_texts = f"{tr}"
                #print('sub', triplet_texts)
                embed_model = self._service_context.embed_model
                embed_outputs = embed_model.get_text_embedding_batch(
                    triplet_texts, show_progress=self._show_progress
                )
                vectors.append((f"vec{tr_i}", embed_outputs[0], {"subject": tr}))
            logger.debug(f"vectors: {vectors}")
            upsert_response = pincone_index.upsert(
                    vectors=vectors,
                    namespace="dexter"
            )
            for triplet in triplets:
                subj, _, obj = triplet
                self.upsert_triplet(triplet)
                index_struct.add_node([subj, obj], n)

            if self.include_embeddings:
                triplet_texts = [str(t) for t in triplets]
                print("geerating", triplet_texts)

                embed_model = self._service_context.embed_model
                embed_outputs = embed_model.get_text_embedding_batch(
                    triplet_texts, show_progress=self._show_progress
                )
                print("embeeding", embed_outputs)

                for rel_text, rel_embed in zip(triplet_texts, embed_outputs):
                    index_struct.add_to_embedding_dict(rel_text, rel_embed)

        return index_struct

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert a document."""
        for n in nodes:
            triplets = self._extract_triplets(
                n.get_content(metadata_mode=MetadataMode.LLM)
            )
            logger.debug(f"Extracted triplets: {triplets}")
            for triplet in triplets:
                subj, _, obj = triplet
                triplet_str = str(triplet)
                self.upsert_triplet(triplet)
                self._index_struct.add_node([subj, obj], n)
                if (
                    self.include_embeddings
                    and triplet_str not in self._index_struct.embedding_dict
                ):
                    rel_embedding = (
                        self._service_context.embed_model.get_text_embedding(
                            triplet_str
                        )
                    )
                    self._index_struct.add_to_embedding_dict(triplet_str, rel_embedding)

    def upsert_triplet(self, triplet: Tuple[str, str, str]) -> None:
        """Insert triplets.

        Used for manual insertion of KG triplets (in the form
        of (subject, relationship, object)).

        Args:
            triplet (str): Knowledge triplet

        """
        print("triplet", triplet)
        self._graph_store.upsert_triplet(*triplet)

    def add_node(self, keywords: List[str], node: BaseNode) -> None:
        """Add node.

        Used for manual insertion of nodes (keyed by keywords).

        Args:
            keywords (List[str]): Keywords to index the node.
            node (Node): Node to be indexed.

        """
        self._index_struct.add_node(keywords, node)
        self._docstore.add_documents([node], allow_update=True)

    def upsert_triplet_and_node(
        self, triplet: Tuple[str, str, str], node: BaseNode
    ) -> None:
        """Upsert KG triplet and node.

        Calls both upsert_triplet and add_node.
        Behavior is idempotent; if Node already exists,
        only triplet will be added.

        Args:
            keywords (List[str]): Keywords to index the node.
            node (Node): Node to be indexed.

        """
        subj, _, obj = triplet
        self.upsert_triplet(triplet)
        self.add_node([subj, obj], node)

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError("Delete is not supported for KG index yet.")

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        node_doc_ids_sets = list(self._index_struct.table.values())
        node_doc_ids = list(set().union(*node_doc_ids_sets))
        nodes = self.docstore.get_nodes(node_doc_ids)

        all_ref_doc_info = {}
        for node in nodes:
            ref_node = node.source_node
            if not ref_node:
                continue

            ref_doc_info = self.docstore.get_ref_doc_info(ref_node.node_id)
            if not ref_doc_info:
                continue

            all_ref_doc_info[ref_node.node_id] = ref_doc_info
        return all_ref_doc_info

    def get_networkx_graph(self, limit: int = 100) -> Any:
        """Get networkx representation of the graph structure.

        Args:
            limit (int): Number of starting nodes to be included in the graph.

        NOTE: This function requires networkx to be installed.
        NOTE: This is a beta feature.

        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "Please install networkx to visualize the graph: `pip install networkx`"
            )

        g = nx.Graph()
        subjs = list(self.index_struct.table.keys())

        # add edges
        rel_map = self._graph_store.get_rel_map(subjs=subjs, depth=1, limit=limit)

        added_nodes = set()
        for keyword in rel_map:
            for path in rel_map[keyword]:
                subj = keyword
                for i in range(0, len(path), 2):
                    if i + 2 >= len(path):
                        break

                    if subj not in added_nodes:
                        g.add_node(subj)
                        added_nodes.add(subj)

                    rel = path[i + 1]
                    obj = path[i + 2]

                    g.add_edge(subj, obj, label=rel, title=rel)
                    subj = obj
        return g

    @property
    def query_context(self) -> Dict[str, Any]:
        return {GRAPH_STORE_KEY: self._graph_store}


# legacy
GPTKnowledgeGraphIndex = KnowledgeGraphIndex
