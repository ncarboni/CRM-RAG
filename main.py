#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application file for the Asinou Dataset Chatbot.
This file contains all the Flask routes and application initialization.
"""

import logging
import os
import sys
import argparse
import shutil
import re
import yaml
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from logging.handlers import RotatingFileHandler
from universal_rag_system import UniversalRagSystem
from config_loader import ConfigLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler("logs/app.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Asinou Dataset Chatbot')
parser.add_argument('--env', type=str, help='Path to environment file')
parser.add_argument('--rebuild', action='store_true', help='Force rebuild of document graph and vector store')
args = parser.parse_args()

# Load configuration
config = ConfigLoader.load_config(args.env)

# Load interface customization from YAML
def load_interface_config():
    """Load interface customization from config/interface.yaml"""
    interface_config_path = 'config/interface.yaml'

    # Default configuration if file doesn't exist
    default_config = {
        'page_title': 'RAG Chat Interface',
        'header_title': 'RAG Chatbot',
        'welcome_message': 'Hello! How can I help you today?',
        'input_placeholder': 'Ask a question...',
        'example_questions': [
            'What is this dataset about?',
            'Tell me about the main entities'
        ],
        'about': {
            'title': 'About This Chat',
            'description': 'This chat interface uses RAG.',
            'features': [],
            'footer': ''
        }
    }

    try:
        if os.path.exists(interface_config_path):
            with open(interface_config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f)
                logger.info(f"Loaded interface configuration from {interface_config_path}")
                return loaded_config if loaded_config else default_config
        else:
            logger.warning(f"Interface config not found at {interface_config_path}, using defaults")
            return default_config
    except Exception as e:
        logger.error(f"Error loading interface config: {str(e)}, using defaults")
        return default_config

interface_config = load_interface_config()

# Flask application setup
app = Flask(__name__)

# Configure Flask secret key for session security
# Even though sessions aren't currently used, this is required for:
# - Future session support
# - CSRF protection
# - Flash messages
# - Secure cookies
import secrets
secret_key = config.get("flask_secret_key")
if not secret_key:
    # Generate a random secret key if not configured
    secret_key = secrets.token_hex(32)
    logger.warning("No FLASK_SECRET_KEY configured in .env, using randomly generated key")
    logger.warning("Sessions will not persist across server restarts. Set FLASK_SECRET_KEY in .env for production")
app.config['SECRET_KEY'] = secret_key

# Initialize the RAG system with configuration
rag_system = UniversalRagSystem(
    endpoint_url=config.get("fuseki_endpoint"),
    config=config
)

# Windows device names that should be blocked (case-insensitive)
WINDOWS_DEVICE_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
    'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
}

def is_safe_path(base_directory, path):
    """
    Validate that a path is safe to serve.
    Prevents:
    1. Path traversal attacks (../)
    2. Absolute paths
    3. Windows device names (CON, AUX, etc.)

    Args:
        base_directory: The base directory to serve from
        path: The requested path

    Returns:
        bool: True if path is safe, False otherwise
    """
    # Reject empty paths
    if not path:
        return False

    # Reject absolute paths
    if os.path.isabs(path):
        logger.warning(f"Rejected absolute path: {path}")
        return False

    # Normalize the path to resolve any .. or . components
    normalized_path = os.path.normpath(path)

    # Check if normalized path tries to escape the base directory
    if normalized_path.startswith('..') or normalized_path.startswith('/'):
        logger.warning(f"Rejected path traversal attempt: {path} (normalized: {normalized_path})")
        return False

    # Check each component of the path for Windows device names
    path_components = Path(normalized_path).parts
    for component in path_components:
        # Extract the base name without extension
        base_name = component.upper().split('.')[0]

        # Check if it's a Windows device name
        if base_name in WINDOWS_DEVICE_NAMES:
            logger.warning(f"Rejected Windows device name in path: {path} (component: {component})")
            return False

    # Verify the final path is within the base directory
    try:
        base_path = Path(base_directory).resolve()
        full_path = (base_path / normalized_path).resolve()

        # Check if the resolved path is within the base directory
        if not str(full_path).startswith(str(base_path)):
            logger.warning(f"Rejected path outside base directory: {path}")
            return False
    except (OSError, ValueError) as e:
        logger.warning(f"Error resolving path {path}: {str(e)}")
        return False

    return True

# Route to serve static files
@app.route('/static/<path:path>')
def send_static(path):
    """
    Serve static files with security validation.
    Prevents path traversal and Windows device name attacks.
    """
    # Validate the path
    if not is_safe_path('static', path):
        logger.warning(f"Blocked unsafe static file request: {path}")
        abort(404)  # Return 404 instead of error message to avoid information disclosure

    return send_from_directory('static', path)

@app.route('/')
def index():
    """Home page route (redirects to chat)"""
    return render_template('chat.html', interface=interface_config)

@app.route('/chat')
def chat():
    """Chat interface route"""
    return render_template('chat.html', interface=interface_config)

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

@app.route('/api/info', methods=['GET'])
def info_api():
    """API endpoint to get system information"""
    # Get dataset description from config or use default
    dataset_description = config.get(
        "dataset_description",
        "Asinou church dataset including frescoes, iconography, and cultural heritage objects"
    )

    return jsonify({
        "llm_provider": config.get("llm_provider", "unknown"),
        "llm_model": config.get("model", "unknown"),
        "embedding_model": config.get("embedding_model", "unknown"),
        "dataset_description": dataset_description,
    })

if __name__ == '__main__':
    # Get port from environment or use default 5001
    port = config.get("port", 5001)

    # Force rebuild if requested
    if args.rebuild:
        logger.info("Rebuilding document graph and vector store...")
        # Delete existing files
        if os.path.exists('data/cache/document_graph.pkl'):
            os.remove('data/cache/document_graph.pkl')
        if os.path.exists('data/cache/vector_index'):
            shutil.rmtree('data/cache/vector_index')

    # Initialize the system
    try:
        # First, check if saved data exists
        if (os.path.exists('data/cache/document_graph.pkl') and os.path.exists('data/cache/vector_index/index.faiss')):
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
                
                if not args.rebuild:  # Only ask for confirmation if not explicitly rebuilding
                    response = input("Do you want to continue with initialization? (y/n): ")
                    if response.lower() != 'y':
                        logger.info("Initialization cancelled. Starting with limited functionality")
                        print("Application starting with limited functionality")
                        app.run(debug=False, host='127.0.0.1', port=port)
                        sys.exit()
            
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
        sys.exit(1)
    
    # Print explicit startup information
    print(f"Starting Flask application...")
    print(f"Running on http://localhost:{port}")
    print(f"Using LLM provider: {config.get('llm_provider')}")
    print(f"Using model: {config.get('model')}")
    print(f"Press CTRL+C to stop the server")
    
    # Run the Flask application
    app.run(debug=False, host='127.0.0.1', port=port)