from fastapi import APIRouter, Depends, HTTPException
from typing import List
from database import DatabaseService, get_db
from models import Environment, EnvironmentCreate
import logging

router = APIRouter(prefix="/environments", tags=["environments"])
logger = logging.getLogger(__name__)


#Create Environment
@router.post("/create", response_model=Environment, status_code=201)
def create_environment(
        environment: EnvironmentCreate,
        db_service: DatabaseService = Depends(DatabaseService)
):
    try:
        environment_id = db_service.create_environment(environment)
        return db_service.get_environment(environment_id)
    except Exception as e:
        logger.error(f"Error creating environment: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))

#Read Environment
@router.get("/{environment_id}", response_model=Environment)
def read_environment(
    environment_id: int,
    db_service: DatabaseService = Depends(DatabaseService)
):
    environment = db_service.get_environment(environment_id)
    if not environment:
        raise HTTPException(status_code=404, detail="Environment not found")

    return environment  # ✅ Return the object directly


#Update Environment
@router.post("/update/{environment_id}", response_model=Environment)
def update_environment(
        environment_id: int,
        environment: EnvironmentCreate,
        db_service: DatabaseService = Depends(DatabaseService)
):
    updated = db_service.update_environment(environment_id, environment)
    if not updated:
        raise HTTPException(status_code=404, detail="Environment not found")
    return updated


#Delete Environment
@router.get("/delete/{environment_id}")
def delete_environment(
        environment_id: int,
        db_service: DatabaseService = Depends(DatabaseService)
):
    if not db_service.delete_environment(environment_id):
        raise HTTPException(status_code=404, detail="Environment not found")
    return {"message": f"Environment with ID {environment_id} deleted successfully."}


#Read All environments
@router.get("/", response_model=List[Environment])
def read_all_environments(db_service: DatabaseService = Depends(DatabaseService)):
    environments = db_service.get_all_environments()
    if not environments:
        raise HTTPException(status_code=404, detail="No environments found")
    return environments


#Read Environments by email.
@router.post("/by-email/", response_model=List[Environment])
def get_all_environments_by_email(
        email: str,
        db_service: DatabaseService = Depends(DatabaseService)
):
    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    environments = db_service.get_environments_by_email(email)
    if not environments:
        raise HTTPException(status_code=404, detail="No environments found")
    return environments
