from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from environment import router as environment_router
from agent import router as agent_router
from dynamic_agents import router as dynamic_agent_router
from run import router as openai_router
from report_gen_fastapi import router as sla_router
from dyn_agent_run import router as dyn_run_router
from database import create_tables
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="AI Agent Management API",
    description="API for managing AI environments and agents",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all routers
app.include_router(environment_router)
app.include_router(agent_router)
app.include_router(dynamic_agent_router)
app.include_router(openai_router)
app.include_router(sla_router)
app.include_router(dyn_run_router)

# Create tables on startup
@app.on_event("startup")
def on_startup():
    create_tables()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1:8000", port=8000)
