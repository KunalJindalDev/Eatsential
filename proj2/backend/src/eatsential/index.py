from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .middleware.jwt_auth import JWTAuthMiddleware
from .middleware.rate_limit import RateLimitMiddleware
from .routers import auth, chat, goals, health, meals, recommend, users, wellness, maps, orders

app = FastAPI()

# Configure Rate Limiting
app.add_middleware(RateLimitMiddleware)

# Configure JWT Authentication
app.add_middleware(JWTAuthMiddleware)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Frontend development server
        "https://eatsential.com",  # Production frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(users.router, prefix="/api")
app.include_router(auth.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(meals.router, prefix="/api")
app.include_router(orders.router, prefix="/api")
app.include_router(goals.router, prefix="/api")
app.include_router(recommend.router, prefix="/api")
app.include_router(wellness.router, prefix="/api")
app.include_router(maps.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/api")
def read_root():
    """Health check endpoint"""
    return {"The server is running": "Hello World"}
