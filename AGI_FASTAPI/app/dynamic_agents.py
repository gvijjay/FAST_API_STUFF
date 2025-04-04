from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import DatabaseService, get_db
from models import DynamicAgent, DynamicAgentCreate
import logging

router = APIRouter(prefix="/dynamic-agents", tags=["dynamic_agents"])
logger = logging.getLogger(__name__)

#Creating Dynamic agents
@router.post("/create", response_model=DynamicAgent, status_code=201)
def create_dynamic_agent(
        dynamic_agent: DynamicAgentCreate,
        db_service: DatabaseService = Depends(DatabaseService)
):
    try:
        agent_id = db_service.create_dynamic_agent(dynamic_agent)  # ✅ Get only the ID
        return db_service.get_dynamic_agent(agent_id)  # ✅ Fetch the full object
    except Exception as e:
        logger.error(f"Error creating dynamic agent: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

#Reading Dynamic agents
@router.get("/{agent_id}", response_model=DynamicAgent)
def read_dynamic_agent(
        agent_id: int,
        db_service: DatabaseService = Depends(DatabaseService)
):
    agent = db_service.get_dynamic_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Dynamic agent not found")
    return agent

#Updating Dynamic agents
@router.put("/update/{agent_id}", response_model=DynamicAgent)
def update_dynamic_agent(
        agent_id: int,
        dynamic_agent: DynamicAgentCreate,
        db_service: DatabaseService = Depends(DatabaseService)
):
    updated = db_service.update_dynamic_agent(agent_id, dynamic_agent)
    if not updated:
        raise HTTPException(status_code=404, detail="Dynamic agent not found")
    return updated

#Deleting Dynamic agents
@router.delete("/delete/{agent_id}")
def delete_dynamic_agent(
        agent_id: int,
        db_service: DatabaseService = Depends(DatabaseService)
):
    if not db_service.delete_dynamic_agent(agent_id):
        raise HTTPException(status_code=404, detail="Dynamic agent not found")
    return {"message": f"Dynamic agent with ID {agent_id} deleted successfully."}


#Reading all the dynamic agents
@router.get("/all", response_model=List[DynamicAgent])
def read_all_dynamic_agents(db_service: DatabaseService = Depends(DatabaseService)):
    agents = db_service.get_all_dynamic_agents()
    if not agents:
        raise HTTPException(status_code=404, detail="No dynamic agents found")
    return agents


#Readin Dynamic agents by mail.
@router.post("/by-email/", response_model=List[DynamicAgent])
def get_all_dynamic_agents_by_email(
        email: str,
        db_service: DatabaseService = Depends(DatabaseService)
):
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    agents = db_service.get_dynamic_agents_by_email(email)
    if not agents:
        raise HTTPException(status_code=404, detail="No dynamic agents found")
    return agents