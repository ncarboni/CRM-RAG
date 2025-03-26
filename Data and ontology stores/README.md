# CRM RAG

## Dual-Store Architecture
The most significant improvement is the creation of a dual-store RAG system that separates RDF data and ontology knowledge:

RDF Vectorstore: Contains factual information about Byzantine art and architecture from the Fuseki database
Ontology Vectorstore: Contains CIDOC-CRM concepts, definitions, and relationships extracted from PDF documentation

This separation allows the system to retrieve both types of knowledge independently and blend them appropriately, giving factual data higher priority while using ontology knowledge to provide deeper context.
