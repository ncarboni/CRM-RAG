#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application file for the Byzantine Art RDF Chatbot.
This file contains all the Flask routes and application initialization.
"""

import logging
import os
from flask import Flask, render_template, request, jsonify, url_for, redirect
from logging.handlers import RotatingFileHandler
from rag_system import FusekiRagSystem
from dotenv import load_dotenv
load_dotenv()  


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("app.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask application setup
app = Flask(__name__)

# Initialize RAG system
ONTOLOGY_DOCS = [
    "docs/CIDOC_CRM_v7.1.3.ttl",  # Path to CIDOC-CRM documentation
    "docs/vir.ttl" # path to VIR ontology
]

# Configure Fuseki endpoint and OpenAI settings
FUSEKI_ENDPOINT = os.environ.get('FUSEKI_ENDPOINT', 'http://localhost:3030/asinou/sparql')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')  # Or 'gpt-4o' if you have access
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.7'))

# Check if API key is set
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable is not set! The application may not function correctly.")

# Initialize the RAG system with configuration
rag_system = FusekiRagSystem(
    endpoint_url=FUSEKI_ENDPOINT,
    openai_api_key=OPENAI_API_KEY,
    openai_model=OPENAI_MODEL,
    temperature=TEMPERATURE,
    ontology_docs_path=ONTOLOGY_DOCS
)

# Replace the @app.before_first_request decorator with a setup function
def initialize_system_startup():
    """Initialize system during startup"""
    try:
        # Test Fuseki connection
        if not rag_system.test_connection():
            logger.warning(f"Failed to connect to Fuseki at {FUSEKI_ENDPOINT}")
            logger.warning("The application will start, but search functionality may be limited")
        
        # Pre-build vector stores for faster responses
        try:
            rag_system.ensure_vectorstores()
            
            # Build additional context indices
            logger.info("Building additional context indices...")
            rag_system.add_geographic_context()
            rag_system.add_temporal_context()
            rag_system.add_iconographic_context()
            logger.info("Context indices built successfully")
        except Exception as e:
            logger.error(f"Error initializing vector stores: {str(e)}")
            logger.info("You can still use the application, but search may not work properly")
    except Exception as e:
        logger.error(f"Error during application initialization: {str(e)}")

@app.route('/')
def index():
    """Home page route"""
    return render_template('index.html')

@app.route('/chat')
def chat():
    """Chat interface route"""
    return render_template('chat.html')

@app.route('/map')
def map_view():
    """Map visualization route"""
    return render_template('map.html')

@app.route('/search')
def search_view():
    """Search interface route"""
    return render_template('search.html')

@app.route('/entity/<path:entity_uri>')
def entity_view(entity_uri):
    """Entity details page"""
    return render_template('entity.html', entity_uri=entity_uri)

@app.route('/ontology')
def ontology_view():
    """Ontology browser route"""
    return render_template('ontology.html')


# API Routes for backend functionality

@app.route('/api/churches')
def get_churches():
    """API endpoint to get all churches with location data"""
    churches = rag_system.get_all_churches()
    return jsonify(churches)

@app.route('/api/entity/<path:entity_uri>')
def get_entity(entity_uri):
    """API endpoint to get entity details"""
    details = rag_system.get_entity_details(entity_uri)
    related = rag_system.get_related_entities(entity_uri)
    
    return jsonify({"details": details, "related": related})

@app.route('/api/entity/<path:entity_uri>/wikidata')
def get_entity_wikidata(entity_uri):
    """API endpoint to get Wikidata information for a specific entity"""
    wikidata_id = rag_system.get_wikidata_for_entity(entity_uri)
    
    if not wikidata_id:
        return jsonify({"error": "No Wikidata ID found for this entity"}), 404
    
    wikidata_info = rag_system.fetch_wikidata_info(wikidata_id)
    
    if not wikidata_info:
        return jsonify({"error": f"Could not fetch Wikidata info for ID: {wikidata_id}"}), 404
    
    return jsonify(wikidata_info)

@app.route('/api/ontology/taxonomy/<taxonomy_group>')
def get_taxonomy_group(taxonomy_group):
    """API endpoint to get entities by taxonomy group"""
    if taxonomy_group not in ["physical_entities", "temporal_entities", "conceptual_entities", "visual_representation"]:
        return jsonify({"error": "Invalid taxonomy group"}), 400
    
    entities = rag_system.get_entity_by_taxonomy_group(taxonomy_group)
    return jsonify(entities)

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat functionality with graph reasoning"""
    question = request.json.get('question', '')
    logger.info(f"Chat request: '{question}'")
    
    if not rag_system.ensure_vectorstores():
        return jsonify({
            "answer": "I'm sorry, I couldn't access the knowledge base. Please try again later.",
            "sources": []
        })
    
    # Get the answer using graph reasoning
    result = rag_system.answer_question_with_graph(question)
    return jsonify(result)

@app.route('/api/search', methods=['POST'])
def search_api():
    """API endpoint for semantic search"""
    query = request.json.get('query', '')
    
    # Make sure vector stores are initialized
    if not rag_system.ensure_vectorstores():
        return jsonify([])
    
    # Execute search
    results = rag_system.search(query, k=5)
    
    # Format results
    formatted_results = []
    for doc in results:
        if doc.metadata.get("source") == "rdf_data":
            entity_uri = doc.metadata.get("entity", "")
            entity_label = doc.metadata.get("label", "")
            
            # Get a shortened excerpt
            content = doc.page_content
            excerpt = content[:200] + "..." if len(content) > 200 else content
            
            formatted_results.append({
                "entity_uri": entity_uri,
                "entity_label": entity_label,
                "excerpt": excerpt,
                "type": "entity"
            })
        elif doc.metadata.get("source") == "ontology_documentation":
            concept_id = doc.metadata.get("concept_id", "")
            concept_name = doc.metadata.get("concept_name", "")
            
            # Get a shortened excerpt
            content = doc.page_content
            excerpt = content[:200] + "..." if len(content) > 200 else content
            
            formatted_results.append({
                "concept_id": concept_id,
                "concept_name": concept_name,
                "excerpt": excerpt,
                "type": "ontology_concept"
            })
    
    return jsonify(formatted_results)

@app.route('/api/wikidata_entities')
def get_wikidata_entities():
    """API endpoint to get all entities with Wikidata references"""
    entities = rag_system.get_wikidata_entities()
    return jsonify(entities)

@app.route('/api/ontology/concept/<concept_id>')
def get_ontology_concept(concept_id):
    """API endpoint to get information about a specific ontology concept"""
    if not rag_system.ontology_processor or not rag_system.ontology_processor.concepts:
        rag_system.ontology_processor.process_ontology_docs()
        
    if not rag_system.ontology_processor.concepts:
        return jsonify({"error": "Ontology not loaded"}), 404
        
    concept = rag_system.ontology_processor.concepts.get(concept_id)
    if not concept:
        return jsonify({"error": f"Concept {concept_id} not found"}), 404
        
    context = rag_system.ontology_processor.get_concept_context(concept_id, depth=1)
    return jsonify(context)

if __name__ == '__main__':
    # Get port from environment or use default 5001
    port = int(os.environ.get('PORT', 5001))
    
    # Check if vector stores exist and try to initialize them
    try:
        rag_system.ensure_vectorstores()
    except Exception as e:
        logger.error(f"Error initializing vector stores: {str(e)}")
        logger.info("Application will still start, but search may not work initially")
    
    # Print explicit startup information
    print(f"Starting Flask application...")
    print(f"Running on http://localhost:{port}")
    print(f"Press CTRL+C to stop the server")
    
    # Run the Flask application
    app.run(debug=True, host='127.0.0.1', port=port)  # Changed host to 127.0.0.1