from rag_system import FusekiRagSystem

def main():
    rag = FusekiRagSystem(
        endpoint_url="http://localhost:3030/asinou/sparql",
        ontology_docs=[
            "docs/CIDOC_CRM_v7.1.3.ttl",
            "docs/vir.ttl"
        ]
    )

    if not rag.test_connection():
        print("SPARQL endpoint not reachable. Check your configuration.")
        return

    print("Building knowledge graph index and ontology...")
    rag.build_graph_index()

    print("\nAsk a question about the Byzantine church. Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        try:
            answer = rag.query_graphrag(user_input)
            print("Bot:", answer)
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
