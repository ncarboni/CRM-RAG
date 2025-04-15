#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application file for the Byzantine Art RDF Chatbot.
This file contains all the Flask routes and application initialization.
"""

import logging
import os
from flask import Flask, render_template, request, jsonify, send_from_directory
from logging.handlers import RotatingFileHandler
from universal_rag_system import UniversalRagSystem
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

# Configure Fuseki endpoint and OpenAI settings
FUSEKI_ENDPOINT = os.environ.get('FUSEKI_ENDPOINT', 'http://localhost:3030/asinou/sparql')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o')
TEMPERATURE = float(os.environ.get('TEMPERATURE', '0.7'))

# Check if API key is set
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable is not set! The application may not function correctly.")

# Initialize the RAG system with configuration
rag_system = UniversalRagSystem(
    endpoint_url=FUSEKI_ENDPOINT,
    openai_api_key=OPENAI_API_KEY,
    openai_model=OPENAI_MODEL,
    temperature=TEMPERATURE
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
    
    wikidata_info = rag_system.fetch_wikidata_info(wikidata_id)
    
    if not wikidata_info:
        return jsonify({"error": f"Could not fetch Wikidata info for ID: {wikidata_id}"}), 404
    
    return jsonify(wikidata_info)

@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat functionality"""
    question = request.json.get('question', '')
    logger.info(f"Chat request: '{question}'")
    
    # Use the direct approach to answer the question
    result = rag_system.answer_question(question)
    return jsonify(result)

if __name__ == '__main__':
    # Get port from environment or use default 5001
    port = int(os.environ.get('PORT', 5001))

    # Initialize the system
    try:
        # First, check if saved data exists
        if (os.path.exists('document_graph.pkl') and os.path.exists('vector_index/index.faiss')):
            logger.info("Found existing data, loading...")
            if rag_system.initialize():
                logger.info("Successfully loaded existing data")
            else:
                raise Exception("Failed to load existing data")
        else:
            # If we need to create new data, check whether it would exceed rate limits
            entities = rag_system.get_all_entities()
            total_entities = len(entities)
            logger.info(f"Need to process {total_entities} entities")
            
            if total_entities > 500:  # Threshold that might cause rate limiting
                logger.warning(f"Large dataset with {total_entities} entities may exceed rate limits")
                logger.warning("Consider running initialization with a higher batch size and longer sleep times")
                response = input("Do you want to continue with initialization? (y/n): ")
                if response.lower() != 'y':
                    logger.info("Initialization cancelled. Starting with limited functionality")
                    print("Application starting with limited functionality")
                    app.run(debug=False, host='127.0.0.1', port=port)
                    exit()
            
            logger.info("Initializing system with new data...")
            if rag_system.initialize():
                logger.info("Successfully initialized with new data")
            else:
                raise Exception("Failed to initialize with new data")
    except Exception as e:
        logger.error(f"Error initializing the system: {str(e)}")
        logger.error("Application cannot start without proper initialization")
        print(f"ERROR: {str(e)}")
        print("Application cannot start without proper initialization. Please fix the issues and try again.")
        exit(1)
    
    # Print explicit startup information
    print(f"Starting Flask application...")
    print(f"Running on http://localhost:{port}")
    print(f"Press CTRL+C to stop the server")
    
    # Run the Flask application
    app.run(debug=False, host='127.0.0.1', port=port)