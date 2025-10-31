#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application file for the Byzantine Art RDF Chatbot (Simple Version).
This version uses the simplified RAG system without GNN for comparison.
"""

import logging
import os
import sys
import argparse
from flask import Flask, render_template, request, jsonify, send_from_directory
from logging.handlers import RotatingFileHandler
from universal_rag_system_simple import UniversalRagSystemSimple
from config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("app_simple.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Byzantine Art RDF Chatbot (Simple Version)')
parser.add_argument('--env', type=str, help='Path to environment file')
parser.add_argument('--rebuild', action='store_true', help='Force rebuild of vector store')
args = parser.parse_args()

# Load configuration
config = ConfigLoader.load_config(args.env)

# Flask application setup
app = Flask(__name__)

# Initialize the simplified RAG system
rag_system = UniversalRagSystemSimple(
    endpoint_url=config.get("fuseki_endpoint"),
    config=config
)

# Route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/')
def index():
    """Home page route (redirects to chat)"""
    return render_template('chat.html')

@app.route('/chat')
def chat():
    """Chat interface route"""
    return render_template('chat.html')

# API endpoint for entity Wikidata information (needed for chat.js)
@app.route('/api/entity/<path:entity_uri>/wikidata')
def get_entity_wikidata(entity_uri):
    """API endpoint to get Wikidata information for a specific entity"""
    wikidata_id = rag_system.get_wikidata_for_entity(entity_uri)
    
    if not wikidata_id:
        return jsonify({"error": "No Wikidata ID found for this entity"}), 404
    
    return jsonify({"wikidata_id": wikidata_id})

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat functionality"""
    question = request.json.get('question', '')
    logger.info(f"Chat request: '{question}'")
    
    # Use the simple approach to answer the question
    result = rag_system.answer_question(question)
    return jsonify(result)

@app.route('/api/info', methods=['GET'])
def info_api():
    """API endpoint to get system information"""
    return jsonify({
        "llm_provider": config.get("llm_provider", "unknown"),
        "llm_model": config.get("model", "unknown"),
        "embedding_model": config.get("embedding_model", "unknown"),
        "system_version": "simple"
    })

if __name__ == '__main__':
    # Get port from environment or use default 5002 (different from main system)
    port = config.get("port", 5002)

    # Force rebuild if requested
    if args.rebuild:
        logger.info("Rebuilding vector store...")
        # Delete existing files
        import shutil
        if os.path.exists('documents_simple.pkl'):
            os.remove('documents_simple.pkl')
        if os.path.exists('vector_index_simple'):
            shutil.rmtree('vector_index_simple')

    # Initialize the system
    try:
        if rag_system.initialize():
            logger.info("Successfully initialized simple RAG system")
        else:
            raise Exception("Failed to initialize simple RAG system")
    except Exception as e:
        logger.error(f"Error initializing the system: {str(e)}")
        logger.error("Application cannot start without proper initialization")
        print(f"ERROR: {str(e)}")
        print("Application cannot start without proper initialization. Please fix the issues and try again.")
        sys.exit(1)
    
    # Print explicit startup information
    print(f"Starting Flask application (SIMPLE VERSION)...")
    print(f"Running on http://localhost:{port}")
    print(f"Using LLM provider: {config.get('llm_provider')}")
    print(f"Using model: {config.get('model')}")
    print(f"System: Simple vector-based retrieval (no GNN, no network metrics)")
    print(f"Press CTRL+C to stop the server")
    
    # Run the Flask application
    app.run(debug=False, host='127.0.0.1', port=port)