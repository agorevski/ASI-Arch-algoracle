#!/bin/bash
# MongoDB API Service Startup Script
set -e
# Color Definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
# Check MongoDB Service Status
check_mongodb() {
    print_info "Checking MongoDB service status..."
    
    # Navigate to the database directory where docker-compose.yml is located
    if ! docker-compose -f docker-compose.yml ps | grep -q "mongodb.*Up"; then
        print_warning "MongoDB service not running, starting..."
        docker-compose -f docker-compose.yml up -d
        sleep 10  # Wait for the containers to start
        
        if docker-compose -f docker-compose.yml ps | grep -q "mongodb.*Up"; then
            print_success "MongoDB service started successfully"
        else
            print_error "MongoDB service failed to start"
            docker-compose -f docker-compose.yml logs mongodb
            exit 1
        fi
    else
        print_success "MongoDB service is running"
    fi
}
# Check Python Dependencies
check_dependencies() {
    print_info "Checking Python dependencies..."
    
    if ! python -c "import fastapi, uvicorn, pymongo, requests" 2>/dev/null; then
        print_warning "Missing Python dependencies, installing..."
        pip install -r requirements.txt
        
        if python -c "import fastapi, uvicorn, pymongo, requests" 2>/dev/null; then
            print_success "Python dependencies installed successfully"
        else
            print_error "Python dependencies installation failed"
            exit 1
        fi
    else
        print_success "Python dependencies are met"
    fi
}
# Start the API Service
start_api() {
    print_info "Starting MongoDB API service..."
    print_info "API Service Address: http://localhost:8001"
    print_info "API Documentation Address: http://localhost:8001/docs"
    print_info "Press Ctrl+C to stop the service"
    
    echo ""
    # The mongodb_api.py is in the current directory (database/)
    python mongodb_api.py
}
# Main function
main() {
    echo "MongoDB API Service Startup Script"
    echo "========================"
    
    # Check MongoDB service
    check_mongodb
    
    # Check Python dependencies
    check_dependencies
    
    # Start API service
    start_api
}
# Run the main function
main "$@"
