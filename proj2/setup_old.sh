#!/usr/bin/env bash

# Eatsential Project Setup Script
# This script automates the setup process for the Eatsential project

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to generate a random JWT secret key
generate_jwt_secret() {
    # Generate a 64-character random hex string (32 bytes)
    if command_exists openssl; then
        openssl rand -hex 32
    else
        # Fallback: use /dev/urandom
        LC_ALL=C tr -dc 'a-zA-Z0-9' < /dev/urandom | head -c 64
    fi
}

# Banner
echo ""
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║        Eatsential Project Setup Script                 ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# Step 1: Check prerequisites
print_info "Checking prerequisites..."

# Check for Bun
if command_exists bun; then
    BUN_VERSION=$(bun --version)
    print_success "Bun is installed (version: $BUN_VERSION)"
else
    print_error "Bun is not installed"
    print_info "Please install Bun by running:"
    echo "  curl -fsSL https://bun.sh/install | bash"
    echo "  Or visit: https://bun.sh"
    exit 1
fi

# Check for uv
if command_exists uv; then
    UV_VERSION=$(uv --version | awk '{print $2}')
    print_success "uv is installed (version: $UV_VERSION)"
else
    print_error "uv is not installed"
    print_info "Please install uv by running:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "  Or visit: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

# Check for Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_success "Python is installed (version: $PYTHON_VERSION)"
else
    print_error "Python 3 is not installed"
    print_info "Please install Python 3.9 or later"
    exit 1
fi

echo ""
print_success "All prerequisites are satisfied!"
echo ""

# Step 2: Confirm installation
print_warning "This script will:"
echo "  1. Install root dependencies"
echo "  2. Install frontend dependencies"
echo "  3. Set up backend Python environment"
echo "  4. Initialize the database with sample data"
echo ""
read -p "Do you want to continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_info "Setup cancelled by user"
    exit 0
fi

echo ""

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Step 3: Install root dependencies
print_info "Installing root dependencies..."
cd "$PROJECT_ROOT"
if bun install; then
    print_success "Root dependencies installed"
else
    print_error "Failed to install root dependencies"
    exit 1
fi

echo ""

# Step 4: Install frontend dependencies
print_info "Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
if bun install; then
    print_success "Frontend dependencies installed"
else
    print_error "Failed to install frontend dependencies"
    exit 1
fi

echo ""

# Step 5: Set up backend Python environment
print_info "Setting up backend Python environment..."
cd "$PROJECT_ROOT/backend"
if uv sync; then
    print_success "Backend dependencies installed"
else
    print_error "Failed to install backend dependencies"
    exit 1
fi

# Install the project in editable mode
print_info "Installing backend project in editable mode..."
if uv pip install -e .; then
    print_success "Backend project installed"
else
    print_error "Failed to install backend project"
    exit 1
fi

echo ""

# Step 6: Initialize database
print_info "Initializing database..."

# Check if .env exists, if not copy from env.example
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        print_info "Copying .env.example to .env..."
        cp .env.example .env
        
        # Generate and replace JWT secret key
        print_info "Generating secure JWT secret key..."
        JWT_SECRET=$(generate_jwt_secret)
        
        # Replace the placeholder JWT secret in .env file
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS version of sed
            sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${JWT_SECRET}/" .env
        else
            # Linux version of sed
            sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=${JWT_SECRET}/" .env
        fi
        
        print_success "Environment configuration created with secure JWT secret"
    else
        print_warning ".env.example not found, skipping .env creation"
    fi
else
    print_info ".env file already exists, skipping..."
fi

# Create database
print_info "Creating database..."
if uv run python scripts/db_initialize/create_init_database.py; then
    print_success "Database created"
else
    print_error "Failed to create database"
    exit 1
fi

# Apply migrations
print_info "Applying database migrations..."
if uv run alembic upgrade head; then
    print_success "Database migrations applied"
else
    print_error "Failed to apply database migrations"
    exit 1
fi

# Seed database
print_info "Seeding database with sample data..."
if uv run python scripts/db_initialize/create_init_database.py --seed; then
    print_success "Database seeded with sample data"
else
    print_error "Failed to seed database"
    exit 1
fi

# Seed restaurants from Google Places
print_info "Seeding restaurants from Google Places API..."
if uv run python scripts/seed_restaurants.py; then
    print_success "Restaurants seeded from Google Places"
else
    print_warning "Failed to seed restaurants (this may be due to API key issues - you can run manually later)"
fi

echo ""

# Seed menu items from CSV data
print_info "Seeding menu items from authentic menu data..."
if uv run python scripts/seed_menus_from_csv.py 2>&1; then
    print_success "Menu items seeded from CSV"
else
    print_warning "Failed to seed menu items from CSV (you can run manually later)"
fi

echo ""

# Step 7: Final success message
echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║            🎉 Setup Complete! 🎉                       ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""
print_success "The Eatsential project has been successfully set up!"
echo "  ⚠️ Please remember to set your Gemini API keys in the backend .env to use the AI features. ⚠️"
echo ""
print_info "Sample credentials:"
echo "  Email:    admin@example.com"
echo "  Password: Admin123!@#"
echo ""
print_info "To start the development servers, run:"
echo "  cd $PROJECT_ROOT"
echo "  bun dev"
echo ""
print_info "This will start:"
echo "  - Frontend at http://localhost:5173"
echo "  - Backend at http://localhost:8000"
echo ""
print_info "Optional: To test email functionality (e.g., email verification):"
echo "  Install MailDev at https://github.com/maildev/maildev"
echo "  Run MailDev"
echo "  View emails at:  http://localhost:1080"
echo ""
print_info "For more information, see INSTALL.md"
echo ""
