#!/usr/bin/env python3
"""Entry point for the CIDOC-CRM RAG system."""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from crm_rag.app import main

if __name__ == '__main__':
    main()
