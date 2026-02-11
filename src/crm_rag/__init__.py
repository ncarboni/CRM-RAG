"""CRM_RAG: Graph-based RAG for CIDOC-CRM knowledge graphs."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/crm_rag → src → Code/

__version__ = "0.5.0"

# Backward-compat aliases so pickles serialized under old module names can load.
from crm_rag import document_store as _ds
sys.modules["graph_document_store"] = _ds
