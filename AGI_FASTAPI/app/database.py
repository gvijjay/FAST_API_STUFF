from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from models import Base, EnvironmentDB, AgentDB, DynamicAgentDB, AgentCreate
from fastapi import Depends
import os
from typing import List, Optional, Type
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    Base.metadata.create_all(bind=engine)


class DatabaseService:
    def __init__(self, db: Session = Depends(get_db)):
        self.db = db

    # Environment CRUD Operations
    def create_environment(self, environment) -> int:
        db_environment = EnvironmentDB(**environment.dict())
        self.db.add(db_environment)
        self.db.commit()
        self.db.refresh(db_environment)
        return db_environment.id  # Return only the ID

    def get_environment(self, environment_id: int) -> Optional[EnvironmentDB]:
        return self.db.query(EnvironmentDB).filter(EnvironmentDB.id == environment_id).first()

    def get_environments_by_email(self, email: str) -> list[Type[EnvironmentDB]]:
        return self.db.query(EnvironmentDB).filter(EnvironmentDB.email == email).all()

    def get_all_environments(self) -> list[Type[EnvironmentDB]]:
        return self.db.query(EnvironmentDB).all()

    def update_environment(self, environment_id: int, environment_update):
        db_environment = self.get_environment(environment_id)
        if db_environment:
            for key, value in environment_update.dict().items():
                setattr(db_environment, key, value)
            self.db.commit()
            self.db.refresh(db_environment)
        return db_environment

    def delete_environment(self, environment_id: int) -> bool:
        db_environment = self.get_environment(environment_id)
        if db_environment:
            self.db.delete(db_environment)
            self.db.commit()
            return True
        return False

    # Agent CRUD Operations
    def create_agent(self, agent: AgentCreate) -> int:
        db_agent = AgentDB(**agent.dict())  # ✅ Convert Pydantic model to ORM model
        self.db.add(db_agent)
        self.db.commit()
        self.db.refresh(db_agent)
        return db_agent.id  # ✅ Return the ID, not the whole object

    def get_agent(self, agent_id: int) -> Optional[AgentDB]:
        return self.db.query(AgentDB).filter(AgentDB.id == agent_id).first()

    def get_agents_by_email(self, email: str) -> list[Type[AgentDB]]:
        return self.db.query(AgentDB).filter(AgentDB.email == email).all()

    def get_all_agents(self) -> list[Type[AgentDB]]:
        return self.db.query(AgentDB).all()

    def update_agent(self, agent_id: int, agent_update):
        db_agent = self.get_agent(agent_id)
        if db_agent:
            for key, value in agent_update.dict().items():
                setattr(db_agent, key, value)
            self.db.commit()
            self.db.refresh(db_agent)
        return db_agent

    def delete_agent(self, agent_id: int) -> bool:
        db_agent = self.get_agent(agent_id)
        if db_agent:
            self.db.delete(db_agent)
            self.db.commit()
            return True
        return False

    # Dynamic Agent CRUD Operations
    def create_dynamic_agent(self, dynamic_agent) -> int:
        db_dynamic_agent = DynamicAgentDB(**dynamic_agent.dict())  # ✅ Convert Pydantic model to ORM model
        self.db.add(db_dynamic_agent)
        self.db.commit()
        self.db.refresh(db_dynamic_agent)
        return db_dynamic_agent.id  # ✅ Return the ID instead of the whole object

    def get_dynamic_agent(self, agent_id: int) -> Optional[DynamicAgentDB]:
        return self.db.query(DynamicAgentDB).filter(DynamicAgentDB.id == agent_id).first()

    def get_dynamic_agents_by_email(self, email: str) -> list[Type[DynamicAgentDB]]:
        return self.db.query(DynamicAgentDB).filter(DynamicAgentDB.email == email).all()

    def get_all_dynamic_agents(self) -> list[Type[DynamicAgentDB]]:
        return self.db.query(DynamicAgentDB).all()

    def update_dynamic_agent(self, agent_id: int, dynamic_agent_update):
        db_dynamic_agent = self.get_dynamic_agent(agent_id)
        if db_dynamic_agent:
            for key, value in dynamic_agent_update.dict().items():
                setattr(db_dynamic_agent, key, value)
            self.db.commit()
            self.db.refresh(db_dynamic_agent)
        return db_dynamic_agent

    def delete_dynamic_agent(self, agent_id: int) -> bool:
        db_dynamic_agent = self.get_dynamic_agent(agent_id)
        if db_dynamic_agent:
            self.db.delete(db_dynamic_agent)
            self.db.commit()
            return True
        return False
