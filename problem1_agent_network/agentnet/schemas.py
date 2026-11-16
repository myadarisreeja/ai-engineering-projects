from typing import List, Optional

from pydantic import BaseModel, Field, HttpUrl


class RegisterAgent(BaseModel):
    company_name: str = Field(min_length=2, max_length=120)
    industry_category: str = Field(min_length=2, max_length=100)
    interest_manifesto: str = Field(min_length=10, max_length=2000)
    webhook_url: Optional[HttpUrl] = None
    topics: List[str] = Field(default_factory=list, description="Optional explicit subscription topics")


class UpdateManifesto(BaseModel):
    interest_manifesto: str = Field(min_length=10, max_length=2000)


class SubscriptionIn(BaseModel):
    topic: str = Field(min_length=2, max_length=60)
    weight: float = Field(default=1.0, gt=0, le=1.0)


class BlockIn(BaseModel):
    blocked_agent_id: str


class PostIn(BaseModel):
    author_agent_id: Optional[str] = None  # optional; must match API key if given
    title: str = Field(min_length=3, max_length=255)
    content: str = Field(min_length=10, max_length=4000)
    source_url: Optional[HttpUrl] = None
    expires_at: Optional[str] = None


class CommentIn(BaseModel):
    author_agent_id: Optional[str] = None
    content: str = Field(min_length=3, max_length=2000)


class ReportIn(BaseModel):
    reason: str = Field(default="", max_length=500)
