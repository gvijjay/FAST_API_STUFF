from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import DatabaseService, get_db
from models import Agent, AgentCreate
import logging

router = APIRouter(prefix="/agents", tags=["agents"])
logger = logging.getLogger(__name__)


#Create an agent
@router.post("/create", response_model=Agent, status_code=201)
def create_agent(
        agent: AgentCreate,
        db_service: DatabaseService = Depends(DatabaseService)
):
    try:
        if not agent.env_id:
            raise HTTPException(status_code=400, detail="Environment ID is required")

        agent_id = db_service.create_agent(agent)  # ✅ Get only the ID
        return db_service.get_agent(agent_id)  # ✅ Use ID to fetch the object
    except Exception as e:
        logger.error(f"Error creating agent: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))


#Reading an agent
@router.get("/{agent_id}", response_model=Agent)
def read_agent(
        agent_id: int,
        db_service: DatabaseService = Depends(DatabaseService)
):
    agent = db_service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    return {
        "id": agent.id,
        "name": agent.name,
        "system_prompt": agent.system_prompt,
        "agent_description": agent.agent_description,
        "backend_id": agent.backend_id,
        "tools": agent.tools,
        "Additional_Features": {"upload_attachment": agent.upload_attachment},
        "env_id": agent.env_id,
        "dynamic_agent_id": agent.dynamic_agent_id,
        "email": agent.email,
        "image_id": agent.image_id
    }

#Updating an agent
@router.put("/update/{agent_id}", response_model=Agent)
def update_agent(
        agent_id: int,
        agent: AgentCreate,
        db_service: DatabaseService = Depends(DatabaseService)
):
    updated = db_service.update_agent(agent_id, agent)
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not found")
    return updated


#Deleting an agent
@router.delete("/delete/{agent_id}")
def delete_agent(
        agent_id: int,
        db_service: DatabaseService = Depends(DatabaseService)
):
    if not db_service.delete_agent(agent_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"message": f"Agent with ID {agent_id} deleted successfully."}


#Reading all the agents
@router.get("/all", response_model=List[Agent])
def read_all_agents(db_service: DatabaseService = Depends(DatabaseService)):
    agents = db_service.get_all_agents()
    if not agents:
        raise HTTPException(status_code=404, detail="No agents found")
    return agents


#Reading an agents by mail
@router.post("/by-email/", response_model=List[Agent])
def get_all_agents_by_email(
        email: str,
        db_service: DatabaseService = Depends(DatabaseService)
):
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    agents = db_service.get_agents_by_email(email)
    if not agents:
        raise HTTPException(status_code=404, detail="No agents found")
    return agents