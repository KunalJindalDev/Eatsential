# How to Run the Application with Feedback Feature

## Prerequisites

The application requires the following tools to be installed:
- **Python 3.13+** (✅ Available)
- **uv** (Python package manager) - Need to install
- **bun** (JavaScript runtime) - Need to install
- **Node.js** (if bun is not available, npm can be used)

## Installation Steps

### Option 1: Automated Setup (Recommended)

```bash
cd proj2
./setup.sh
```

This script will:
- Check for required tools
- Install all dependencies
- Set up the database
- Generate necessary configuration files

### Option 2: Manual Setup

#### 1. Install uv (Python package manager)
```bash
# macOS
brew install uv

# Or using pip
pip install uv
```

#### 2. Install bun (JavaScript runtime)
```bash
# macOS
curl -fsSL https://bun.sh/install | bash

# Or use npm/npx
npm install -g bun
```

#### 3. Install Backend Dependencies
```bash
cd proj2/backend
uv sync
```

#### 4. Install Frontend Dependencies
```bash
cd proj2/frontend
bun install
# Or if bun is not available:
npm install
```

#### 5. Apply Database Migration
```bash
cd proj2/backend
uv run alembic upgrade head
```

#### 6. Set Up Environment Variables
```bash
cd proj2/backend
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY if needed
```

## Running the Application

### Start Both Frontend and Backend
```bash
cd proj2
bun dev
```

This will start:
- **Backend**: http://localhost:8000
- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

### Start Backend Only
```bash
cd proj2/backend
uv run fastapi dev src/eatsential/index.py
# Or
uv run uvicorn src.eatsential.index:app --reload
```

### Start Frontend Only
```bash
cd proj2/frontend
bun run dev
# Or if using npm:
npm run dev
```

## Testing the Feedback Feature

Once the application is running:

1. **Navigate to the recommendations page** in your browser
   - Go to http://localhost:5173
   - Log in with credentials (e.g., admin@example.com / Admin123!@#)

2. **View recommendations**
   - The recommendation cards will now show Like/Dislike buttons

3. **Submit feedback**
   - Click "Like" or "Dislike" on any recommendation
   - The feedback will be saved to the database

4. **See the feedback in action**
   - Request new recommendations
   - Disliked items will be filtered out
   - Liked items will appear with boosted scores

5. **Check the API**
   - Visit http://localhost:8000/docs
   - Look for the `/api/recommend/feedback` endpoint
   - Test it directly from the Swagger UI

## Verification

### Check Database Migration
```bash
cd proj2/backend
uv run alembic current
# Should show: 013_add_recommendation_feedback_table
```

### Check API Endpoint
```bash
curl -X POST http://localhost:8000/api/recommend/feedback \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "item_id": "test-item-id",
    "item_type": "meal",
    "feedback_type": "like"
  }'
```

## Troubleshooting

### If `uv` command not found:
```bash
# Install uv
pip install uv
# Or
brew install uv
```

### If `bun` command not found:
```bash
# Install bun
curl -fsSL https://bun.sh/install | bash
# Or use npm instead
npm install
npm run dev
```

### If database migration fails:
```bash
cd proj2/backend
# Check current migration status
uv run alembic current
# If needed, upgrade
uv run alembic upgrade head
```

### If port already in use:
- Backend: Change port in `uvicorn` command or use `--port 8001`
- Frontend: Vite will automatically use the next available port

## What Changed

The feedback feature adds:
- ✅ Like/Dislike buttons on recommendation cards
- ✅ New API endpoint: `POST /api/recommend/feedback`
- ✅ Database table: `recommendation_feedback`
- ✅ Automatic filtering of disliked items
- ✅ Score boosting for liked items

All changes are documented in `FEEDBACK_FEATURE_CHANGES.md`

