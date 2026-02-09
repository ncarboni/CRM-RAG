#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application file for the CIDOC-CRM RAG system.
This file contains all the Flask routes and application initialization.
"""

# Fix OpenMP conflict between PyTorch and faiss-cpu on macOS
# Must be set before any imports that load these libraries
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import sys
import argparse
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from logging.handlers import RotatingFileHandler
from config_loader import ConfigLoader
from dataset_manager import DatasetManager

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
parser = argparse.ArgumentParser(description='CIDOC-CRM RAG System')
parser.add_argument('--env', type=str, help='Path to environment file')
parser.add_argument('--rebuild', action='store_true', help='Force rebuild of document graph and vector store')
parser.add_argument('--embedding-provider', type=str, default=None,
                    choices=['openai', 'sentence-transformers', 'local', 'ollama'],
                    help='Embedding provider to use. "local" or "sentence-transformers" uses local embeddings (fast, no API). Default: same as LLM provider.')
parser.add_argument('--embedding-model', type=str, default=None,
                    help='Embedding model name. For sentence-transformers: "BAAI/bge-m3" (default), "all-MiniLM-L6-v2" (fast), etc.')
parser.add_argument('--no-embedding-cache', action='store_true',
                    help='Disable embedding cache (force re-embedding all documents)')
parser.add_argument('--dataset', type=str, default=None,
                    help='Dataset ID to process (from datasets.yaml). Use with --rebuild to process a specific dataset.')
parser.add_argument('--debug', action='store_true',
                    help='Enable debug logging for detailed traversal tracing.')
args = parser.parse_args()

# Enable debug logging if requested
if args.debug:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('universal_rag_system').setLevel(logging.DEBUG)
    logger.info("Debug logging enabled - will trace event-aware traversal decisions")

# Load configuration
config = ConfigLoader.load_config(args.env)

# Pass embedding provider config if specified via CLI
if args.embedding_provider:
    config['embedding_provider'] = args.embedding_provider
    logger.info(f"Using embedding provider from CLI: {args.embedding_provider}")

    # Set default embedding model for local provider if not explicitly specified
    if args.embedding_provider in ('local', 'sentence-transformers') and not args.embedding_model:
        # Don't use OpenAI model names with sentence-transformers
        current_model = config.get('embedding_model', '')
        if current_model.startswith('text-embedding') or 'openai' in current_model.lower():
            config['embedding_model'] = 'BAAI/bge-m3'
            logger.info(f"Using default sentence-transformers model: BAAI/bge-m3")

if args.embedding_model:
    config['embedding_model'] = args.embedding_model
    logger.info(f"Using embedding model from CLI: {args.embedding_model}")
if args.no_embedding_cache:
    config['use_embedding_cache'] = False
    logger.info("Embedding cache disabled via CLI")

interface_config = ConfigLoader.load_interface_config()

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

# Load datasets configuration
datasets_config = ConfigLoader.load_datasets_config()

# Require datasets.yaml configuration
if not datasets_config.get('datasets'):
    logger.error("No datasets configured. Please create config/datasets.yaml with at least one dataset.")
    print("ERROR: No datasets configured.")
    print("Please create config/datasets.yaml with your SPARQL endpoints.")
    print("See config/datasets.yaml.example or README.md for configuration instructions.")
    sys.exit(1)

# Use DatasetManager for lazy loading of datasets
logger.info(f"Found {len(datasets_config['datasets'])} datasets in configuration")
dataset_manager = DatasetManager(datasets_config, config)

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
    # Get dataset_id from query parameter (use default if not specified)
    dataset_id = request.args.get('dataset_id') or datasets_config.get('default_dataset')

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400

    try:
        current_rag = dataset_manager.get_dataset(dataset_id)
    except (ValueError, RuntimeError):
        return jsonify({"error": "Dataset not available"}), 500

    wikidata_id = current_rag.get_wikidata_for_entity(entity_uri)

    if not wikidata_id:
        return jsonify({"error": "No Wikidata ID found for this entity"}), 404

    wikidata_info = current_rag.fetch_wikidata_info(wikidata_id)

    if not wikidata_info:
        return jsonify({"error": f"Could not fetch Wikidata info for ID: {wikidata_id}"}), 404

    return jsonify(wikidata_info)

@app.route('/api/datasets', methods=['GET'])
def list_datasets():
    """API endpoint to list available datasets"""
    return jsonify({
        "datasets": dataset_manager.list_datasets(),
        "default": datasets_config.get('default_dataset')
    })


@app.route('/api/datasets/<dataset_id>/select', methods=['POST'])
def select_dataset(dataset_id):
    """API endpoint to initialize and select a dataset, returns interface config"""
    try:
        # This will lazy-load the dataset if not already loaded
        dataset_manager.get_dataset(dataset_id)
        # Get merged interface config
        merged_interface = dataset_manager.get_interface_config(dataset_id, interface_config)
        return jsonify({
            "success": True,
            "interface": merged_interface,
            "initialized": True
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/chat', methods=['POST'])
def chat_api():
    """API endpoint for chat functionality"""
    question = request.json.get('question', '')
    dataset_id = request.json.get('dataset_id')
    logger.info(f"Chat request: '{question}' (dataset: {dataset_id})")

    if not dataset_id:
        return jsonify({"error": "dataset_id is required"}), 400

    try:
        current_rag = dataset_manager.get_dataset(dataset_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500

    # Pass conversation history for follow-up context
    chat_history = request.json.get('chat_history', None)
    result = current_rag.answer_question(question, chat_history=chat_history)
    return jsonify(result)

@app.route('/api/info', methods=['GET'])
def info_api():
    """API endpoint to get system information"""
    return jsonify({
        "llm_provider": config.get("llm_provider", "unknown"),
        "llm_model": config.get("model", "unknown"),
        "embedding_model": config.get("embedding_model", "unknown"),
        "datasets_count": len(datasets_config.get('datasets', {})),
        "default_dataset": datasets_config.get('default_dataset'),
    })

if __name__ == '__main__':
    # Get port from environment or use default 5001
    port = config.get("port", 5001)

    # Determine which dataset to process
    target_dataset = args.dataset
    if target_dataset:
        # Validate dataset exists
        available_datasets = list(datasets_config.get('datasets', {}).keys())
        if target_dataset not in available_datasets:
            print(f"ERROR: Dataset '{target_dataset}' not found.")
            print(f"Available datasets: {', '.join(available_datasets)}")
            sys.exit(1)
        logger.info(f"Target dataset specified: {target_dataset}")

    # Handle rebuild mode
    if args.rebuild:
        logger.info("Rebuilding document graph and vector store...")
        rebuild_ds = target_dataset or datasets_config.get('default_dataset')
        if rebuild_ds:
            cache_paths = dataset_manager.get_cache_paths(rebuild_ds)
            if os.path.exists(cache_paths['document_graph']):
                os.remove(cache_paths['document_graph'])
            if os.path.exists(cache_paths['vector_index_dir']):
                shutil.rmtree(cache_paths['vector_index_dir'])
            # Also clear embedding cache if it exists
            embedding_cache_dir = os.path.join(cache_paths['cache_dir'], 'embeddings')
            if os.path.exists(embedding_cache_dir):
                shutil.rmtree(embedding_cache_dir)
            logger.info(f"Cleared cache for dataset: {rebuild_ds}")
        else:
            logger.warning("No dataset specified and no default_dataset configured")

    # Override default dataset if --dataset was specified
    if target_dataset:
        datasets_config['default_dataset'] = target_dataset
        logger.info(f"Default dataset overridden to: {target_dataset}")

    # Initialize the system
    try:
        if target_dataset:
            # Process specific dataset
            logger.info(f"Processing dataset: {target_dataset}")
            dataset_manager.get_dataset(target_dataset)
            logger.info(f"Successfully processed dataset: {target_dataset}")

        else:
            # Lazy loading, no initialization needed at startup
            logger.info("Datasets will be loaded on first access")
            # Optionally pre-load the default dataset if it has cache
            default_ds = datasets_config.get('default_dataset')
            if default_ds and not args.rebuild:
                cache_paths = dataset_manager.get_cache_paths(default_ds)
                if os.path.exists(cache_paths['document_graph']) and os.path.exists(cache_paths['vector_index_dir']):
                    logger.info(f"Pre-loading default dataset: {default_ds}")
                    try:
                        dataset_manager.get_dataset(default_ds)
                        logger.info(f"Successfully loaded default dataset: {default_ds}")
                    except Exception as e:
                        logger.warning(f"Could not pre-load default dataset: {str(e)}")
                        logger.info("Dataset will be loaded on first access")
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
    print(f"Available datasets: {len(datasets_config.get('datasets', {}))}")
    print(f"Press CTRL+C to stop the server")

    # Run the Flask application
    app.run(debug=False, host='127.0.0.1', port=port)