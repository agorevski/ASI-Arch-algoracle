#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Web API Interface for RAG Service
Provides HTTP REST API based on Flask.
"""
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import traceback
from rag_service import OpenSearchRAGService
import sys
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from config_loader import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(name)s-%(levelname)s-%(message)s')

# Create Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS)
# Global RAG service instance
rag_service = None


def init_rag_service(data_dir: str):
    """Initialize the RAG service with documents from the specified directory.

    Loads cognition data from the given directory and indexes the documents
    into the OpenSearch RAG service.

    Args:
        data_dir: Path to the directory containing cognition data files.

    Returns:
        bool: True if initialization and indexing succeeded, False otherwise.
    """
    global rag_service
    start_time = time.time()
    logging.info(f"[CALL] init_rag_service - data_dir: {data_dir}")

    try:
        logging.info("Initializing RAG service...")
        rag_service = OpenSearchRAGService()

        # Load and index data
        documents = rag_service.load_cognition_data(data_dir=data_dir)
        if documents:
            success = rag_service.index_documents(documents)
            if success:
                duration_ms = (time.time() - start_time) * 1000
                logging.info(f"[EXIT] init_rag_service - Success: True, Documents: {len(documents)}, Duration: {duration_ms:.2f}ms")
                logging.info("RAG service initialization successful")
                return True
            else:
                duration_ms = (time.time() - start_time) * 1000
                logging.error(f"[EXIT] init_rag_service - Success: False, Reason: Document indexing failed, Duration: {duration_ms:.2f}ms")
                logging.error("Document indexing failed")
                return False
        else:
            duration_ms = (time.time() - start_time) * 1000
            logging.error(f"[EXIT] init_rag_service - Success: False, Reason: No documents loaded, Duration: {duration_ms:.2f}ms")
            logging.error("No documents loaded")
            return False

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(f"[EXIT] init_rag_service - Success: False, Error: {str(e)}, Duration: {duration_ms:.2f}ms")
        logging.error(f"Error initializing RAG service: {e}")
        logging.error(traceback.format_exc())
        return False

# Note: The before_first_request decorator has been removed in Flask 2.2+
# The service is now initialized manually in the main function


@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler for unhandled errors.

    Logs detailed error information including exception type, message,
    request details, and full traceback.

    Args:
        e: The exception that was raised.

    Returns:
        tuple: A JSON response containing error details and HTTP status 500.
    """
    logging.error("[CALL] handle_exception - Unhandled exception occurred")
    logging.error(f"Exception Type: {type(e).__name__}")
    logging.error(f"Exception Message: {str(e)}")
    logging.error(f"Request Method: {request.method}")
    logging.error(f"Request Path: {request.path}")
    logging.error(f"Request IP: {request.remote_addr}")
    logging.error(f"Traceback:\n{traceback.format_exc()}")
    logging.error("[EXIT] handle_exception - Status: 500")

    return jsonify({
        "error": "Internal Server Error",
        "message": str(e)
    }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify service status.

    Checks if the RAG service is initialized and retrieves current statistics.

    Returns:
        tuple: A JSON response containing health status and service stats.
            Returns HTTP 200 if healthy, 503 if service unavailable.
    """
    start_time = time.time()
    logging.info(f"[CALL] health_check - IP: {request.remote_addr}")

    if rag_service is None:
        duration_ms = (time.time() - start_time) * 1000
        logging.warning(f"[EXIT] health_check - Status: 503, Reason: RAG service not initialized, Duration: {duration_ms:.2f}ms")
        return jsonify({
            "status": "error",
            "message": "RAG service not initialized"
        }), 503

    try:
        stats = rag_service.get_stats()
        duration_ms = (time.time() - start_time) * 1000
        logging.info(f"[EXIT] health_check - Status: 200, Duration: {duration_ms:.2f}ms")
        return jsonify({
            "status": "healthy",
            "service": "RAG API",
            "stats": stats
        })
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(f"[EXIT] health_check - Status: 503, Error: {str(e)}, Duration: {duration_ms:.2f}ms")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 503


@app.route('/search', methods=['POST'])
def search_patterns():
    """Search for similar experiment trigger patterns.

    Accepts a POST request with a JSON body containing search parameters
    and returns matching patterns from the indexed documents.

    Request Body:
        query (str): The search query text.
        k (int, optional): Number of results to return. Defaults to 5. Max 50.
        similarity_threshold (float, optional): Minimum similarity score (0-1).
            Defaults to 0.6.

    Returns:
        tuple: A JSON response containing the query, total results count,
            and list of matching results. Returns HTTP 200 on success,
            400 for invalid parameters, 500 on error, or 503 if service unavailable.
    """
    start_time = time.time()
    logging.info(f"[CALL] search_patterns - IP: {request.remote_addr}")

    if rag_service is None:
        duration_ms = (time.time() - start_time) * 1000
        logging.warning(f"[EXIT] search_patterns - Status: 503, Reason: RAG service not initialized, Duration: {duration_ms:.2f}ms")
        return jsonify({
            "error": "RAG service not initialized"
        }), 503

    try:
        data = request.get_json()
        logging.info(f"Search request received: {data}")
        if not data:
            duration_ms = (time.time() - start_time) * 1000
            logging.warning(f"[EXIT] search_patterns - Status: 400, Reason: Empty request body, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "error": "Request body cannot be empty"
            }), 400

        query = data.get('query', '').strip()
        if not query:
            duration_ms = (time.time() - start_time) * 1000
            logging.warning(f"[EXIT] search_patterns - Status: 400, Reason: Empty query, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "error": "Query parameter cannot be empty"
            }), 400

        k = data.get('k', 5)
        similarity_threshold = data.get('similarity_threshold', 0.6)

        # Validate parameters
        if not isinstance(k, int) or k <= 0 or k > 50:
            duration_ms = (time.time() - start_time) * 1000
            logging.warning(f"[EXIT] search_patterns - Status: 400, Reason: Invalid k value: {k}, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "error": "Parameter k must be an integer between 1 and 50"
            }), 400

        if not isinstance(similarity_threshold, (int, float)) or similarity_threshold < 0 or similarity_threshold > 1:
            duration_ms = (time.time() - start_time) * 1000
            logging.warning(f"[EXIT] search_patterns - Status: 400, Reason: Invalid similarity_threshold: {similarity_threshold}, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "error": "Similarity threshold must be a value between 0 and 1"
            }), 400

        # Perform the search
        results = rag_service.search_similar_patterns(
            query=query,
            k=k,
            similarity_threshold=similarity_threshold
        )

        duration_ms = (time.time() - start_time) * 1000
        logging.info(f"[EXIT] search_patterns - Status: 200, Query: '{query[:50]}...', Results: {len(results)}, k: {k}, Threshold: {similarity_threshold}, Duration: {duration_ms:.2f}ms")

        return jsonify({
            "query": query,
            "total_results": len(results),
            "results": results
        })

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(f"[EXIT] search_patterns - Status: 500, Error: {str(e)}, Duration: {duration_ms:.2f}ms")
        logging.error(f"Error during search: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            "error": "Search failed",
            "message": str(e)
        }), 500


@app.route('/paper/<paper_key>', methods=['GET'])
def get_paper_documents(paper_key):
    """Get all related documents based on the paper key.

    Retrieves all indexed documents associated with a specific paper identifier.

    Args:
        paper_key: The unique identifier for the paper to search for.

    Returns:
        tuple: A JSON response containing the paper key, total document count,
            and list of documents. Returns HTTP 200 on success, 400 for empty key,
            500 on error, or 503 if service unavailable.
    """
    start_time = time.time()
    logging.info(f"[CALL] get_paper_documents - IP: {request.remote_addr}, paper_key: {paper_key}")

    if rag_service is None:
        duration_ms = (time.time() - start_time) * 1000
        logging.warning(f"[EXIT] get_paper_documents - Status: 503, Reason: RAG service not initialized, Duration: {duration_ms:.2f}ms")
        return jsonify({
            "error": "RAG service not initialized"
        }), 503

    try:
        if not paper_key.strip():
            duration_ms = (time.time() - start_time) * 1000
            logging.warning(f"[EXIT] get_paper_documents - Status: 400, Reason: Empty paper key, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "error": "Paper key cannot be empty"
            }), 400

        results = rag_service.get_document_by_paper(paper_key)

        duration_ms = (time.time() - start_time) * 1000
        logging.info(f"[EXIT] get_paper_documents - Status: 200, paper_key: {paper_key}, Documents: {len(results)}, Duration: {duration_ms:.2f}ms")

        return jsonify({
            "paper_key": paper_key,
            "total_documents": len(results),
            "documents": results
        })

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(f"[EXIT] get_paper_documents - Status: 500, Error: {str(e)}, Duration: {duration_ms:.2f}ms")
        logging.error(f"Error during paper key search: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            "error": "Search failed",
            "message": str(e)
        }), 500


@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get index statistics from the RAG service.

    Retrieves current statistics about the indexed documents and service state.

    Returns:
        tuple: A JSON response containing index statistics.
            Returns HTTP 200 on success, 500 on error, or 503 if service unavailable.
    """
    start_time = time.time()
    logging.info(f"[CALL] get_statistics - IP: {request.remote_addr}")

    if rag_service is None:
        duration_ms = (time.time() - start_time) * 1000
        logging.warning(f"[EXIT] get_statistics - Status: 503, Reason: RAG service not initialized, Duration: {duration_ms:.2f}ms")
        return jsonify({
            "error": "RAG service not initialized"
        }), 503

    try:
        stats = rag_service.get_stats()
        duration_ms = (time.time() - start_time) * 1000
        logging.info(f"[EXIT] get_statistics - Status: 200, Duration: {duration_ms:.2f}ms")
        return jsonify(stats)

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(f"[EXIT] get_statistics - Status: 500, Error: {str(e)}, Duration: {duration_ms:.2f}ms")
        logging.error(f"Error getting statistics: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            "error": "Failed to get statistics",
            "message": str(e)
        }), 500


@app.route('/reinit', methods=['POST'])
def reinitialize_service():
    """Re-initialize the RAG service.

    Reloads cognition data from the configured directory and re-indexes
    all documents. Useful for refreshing the service after data updates.

    Returns:
        tuple: A JSON response indicating success or failure status.
            Returns HTTP 200 on success, or 500 on error.
    """
    start_time = time.time()
    logging.info(f"[CALL] reinitialize_service - IP: {request.remote_addr}")

    try:
        data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'settings', 'architecture', Config.ARCHITECTURE, 'cognition')
        success = init_rag_service(data_dir=data_directory)
        if success:
            duration_ms = (time.time() - start_time) * 1000
            logging.info(f"[EXIT] reinitialize_service - Status: 200, Success: True, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "status": "success",
                "message": "RAG service re-initialized successfully"
            })
        else:
            duration_ms = (time.time() - start_time) * 1000
            logging.error(f"[EXIT] reinitialize_service - Status: 500, Success: False, Duration: {duration_ms:.2f}ms")
            return jsonify({
                "status": "error",
                "message": "RAG service re-initialization failed"
            }), 500

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logging.error(f"[EXIT] reinitialize_service - Status: 500, Error: {str(e)}, Duration: {duration_ms:.2f}ms")
        logging.error(f"Error during re-initialization: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/', methods=['GET'])
def api_info():
    """API information and usage instructions.

    Provides documentation about available endpoints, their methods,
    and example usage for the RAG API service.

    Returns:
        Response: A JSON response containing service description,
            available endpoints, and a search example.
    """
    start_time = time.time()
    logging.info(f"[CALL] api_info - IP: {request.remote_addr}")

    duration_ms = (time.time() - start_time) * 1000
    logging.info(f"[EXIT] api_info - Status: 200, Duration: {duration_ms:.2f}ms")

    return jsonify({
        "service": "RAG API for Cognition Database",
        "description": "RAG service based on OpenSearch for searching and retrieving experimental trigger patterns",
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Health check",
            "GET /stats": "Get index statistics",
            "POST /search": "Search for similar experiment trigger patterns",
            "GET /paper/<paper_key>": "Get documents by paper key",
            "POST /reinit": "Re-initialize service"
        },
        "search_example": {
            "url": "/search",
            "method": "POST",
            "body": {
                "query": "The model performs poorly on long sequences",
                "k": 5,
                "similarity_threshold": 0.6
            }
        }
    })


if __name__ == '__main__':
    logging.info("Starting RAG API service...")
    logging.info("Initializing RAG Service...")

    data_directory = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'settings', 'architecture', Config.ARCHITECTURE, 'cognition')
    # Initialize the RAG service before starting the Flask application
    success = init_rag_service(data_dir=data_directory)
    if not success:
        logging.warning("❌ RAG service initialization failed, please check the logs")
        exit(1)

    logging.info("✅ RAG service initialized successfully")
    logging.info("API Documentation: http://localhost:5000/")
    logging.info("Health Check: http://localhost:5000/health")
    logging.info("Statistics: http://localhost:5000/stats")

    app.run(
        host='0.0.0.0',
        port=13142,
        debug=False,
        threaded=True
    )
