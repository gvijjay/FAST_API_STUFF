from sqlalchemy import Column, Integer, String, Text, Numeric, Boolean, ForeignKey
from sqlalchemy.orm import relationship, declarative_base
from pydantic import BaseModel
from typing import Optional, List

Base = declarative_base()


# --------------------------
# Pydantic Schemas (for API)
# --------------------------

class EnvironmentBase(BaseModel):
    name: str
    api_key: str
    model: str
    temperature: float = 0.5
    email: Optional[str] = None


class EnvironmentCreate(EnvironmentBase):
    pass


class Environment(EnvironmentBase):
    id: int

    class Config:
        from_attributes = True


class AgentBase(BaseModel):
    name: str
    system_prompt: Optional[str] = None
    agent_description: Optional[str] = None
    backend_id: Optional[str] = None
    tools: Optional[str] = None
    upload_attachment: bool = False
    env_id: Optional[int] = None
    dynamic_agent_id: Optional[int] = None
    email: Optional[str] = None
    image_id: Optional[int] = None


class AgentCreate(AgentBase):
    pass


class Agent(AgentBase):
    id: int

    class Config:
        from_attributes = True


class DynamicAgentBase(BaseModel):
    agent_name: str
    agent_goal: Optional[str] = None
    agent_description: Optional[str] = None
    agent_instruction: Optional[str] = None
    email: Optional[str] = None


class DynamicAgentCreate(DynamicAgentBase):
    pass


class DynamicAgent(DynamicAgentBase):
    id: int

    class Config:
        from_attributes = True


# --------------------------
# SQLAlchemy Models (for DB) Column names present
# --------------------------

class EnvironmentDB(Base):
    __tablename__ = "environment"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    api_key = Column(Text, nullable=False)
    model = Column(String(100), nullable=False)
    temperature = Column(Numeric(3, 2), default=0.5)
    email = Column(String(255))

    agents = relationship("AgentDB", back_populates="environment")


class AgentDB(Base):
    __tablename__ = "ai_all_agents"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    system_prompt = Column(Text)
    agent_description = Column(Text)
    backend_id = Column(Text)
    tools = Column(Text)
    upload_attachment = Column(Boolean, default=False)
    env_id = Column(Integer, ForeignKey("environment.id"))
    dynamic_agent_id = Column(Integer, ForeignKey("dynamic_ai_agents.id"))
    email = Column(String(255))
    image_id = Column(Integer)

    environment = relationship("EnvironmentDB", back_populates="agents")
    dynamic_agent = relationship("DynamicAgentDB", back_populates="agents")


class DynamicAgentDB(Base):
    __tablename__ = "dynamic_ai_agents"

    id = Column(Integer, primary_key=True, index=True)
    agent_name = Column(String(100), nullable=False)
    agent_goal = Column(Text)
    agent_description = Column(Text)
    agent_instruction = Column(Text)
    email = Column(String(255))

    agents = relationship("AgentDB", back_populates="dynamic_agent")
