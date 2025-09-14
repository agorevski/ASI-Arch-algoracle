class Config:
    """Configuration settings for the experiment."""
    # Target file
    SOURCE_FILE: str = "evolve file"

    # Training script
    BASH_SCRIPT: str = "your training script"

    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"

    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"

    # Code pool directory
    CODE_POOL: str = "./pool"

    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 5

    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 20

    # RAG service URL
    RAG: str = "your rag url"

    # Database URL
    DATABASE: str = "your databse url"

    # OpenAI API configuration
    AZURE_ENDPOINT: str = "https://endpoint_here.openai.azure.com/"
    AZURE_DEPLOYMENT: str = "deployment_here"
    API_VERSION: str = "2025-01-01-preview"
    API_KEY: str = "api_key_kere"
