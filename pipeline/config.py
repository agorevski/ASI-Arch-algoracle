class Config:
    """Configuration settings for the experiment."""

    # Training script (should be in the /pipeline folder)
    TRAINING_SCRIPT: str = "train.py"

    # Experiment results
    RESULT_FILE: str = "./files/analysis/loss.csv"
    RESULT_FILE_TEST: str = "./files/analysis/benchmark.csv"

    # Debug file
    DEBUG_FILE: str = "./files/debug/training_error.txt"

    # Code pool directory
    CODE_POOL: str = "./pool/deltanet"

    # Target file
    SOURCE_FILE: str = f"{CODE_POOL}/deltanet_base.py"

    # Cognition directory
    COGNITION_DIR: str = "cognition/linear_attention"

    # Maximum number of debug attempts
    MAX_DEBUG_ATTEMPT: int = 5

    # Maximum number of retry attempts
    MAX_RETRY_ATTEMPTS: int = 20

    # RAG service URL
    RAG: str = "http://127.0.0.1:13124"

    # Database URL
    DATABASE: str = "http://0.0.0.0:8001"

    # OpenAI API configuration
    AZURE_ENDPOINT: str = "https://your_endpoint.openai.azure.com/"
    AZURE_DEPLOYMENT: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_JUDGER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_ANALYZER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_CHECKER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_EVOLVER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_PLANNER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_SUMMARIZER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_DEBUGGER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_TRAINER: str = "gpt-5"
    AZURE_DEPLOYMENT_MODEL_MOTIVATION_CHECKER: str = "gpt-5"
    AZURE_DEPLOYMENT_RAG_GENERATION: str = "gpt-5"
    API_VERSION: str = "2025-01-01-preview"
    API_KEY: str = "your_key"

    # AML Configurations
    AML_SUBSCRIPTION_ID: str = "your_subscription_id"
    AML_RESOURCE_GROUP: str = "your_resource_group"
    AML_WORKSPACE_NAME: str = "your_workspace_name"
    AML_CLUSTER_NAME: str = "your_cluster_name"
