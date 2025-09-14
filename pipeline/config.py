class Config:
    """Configuration settings for the experiment."""
    # Target file
    SOURCE_FILE: str = "./pool/deltanet_base.py"

    # Training script (use train_simple.py for testing without PyTorch)
    BASH_SCRIPT: str = "python train.py"

    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"

    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"

    # Code pool directory
    CODE_POOL: str = "./pool"

    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 3

    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 20

    # RAG service URL
    RAG: str = "http://127.0.0.1:13124"

    # Database URL
    DATABASE: str = "http://0.0.0.0:8001"

    # OpenAI API configuration
    AZURE_ENDPOINT: str = "https://your_endpoint.openai.azure.com/"
    AZURE_DEPLOYMENT: str = "your_deployment"
    API_VERSION: str = "2025-01-01-preview"
    API_KEY: str = "your_key"
