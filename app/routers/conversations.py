import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.events.stream import event_bus
from app.models.db_models import Conversation
from app.models.schemas import ConversationCreate, ConversationResponse, PipelineEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["Conversations"])


def _conv_to_response(conv: Conversation) -> dict[str, Any]:
    data = conv.data or {}
    return {
        "id": conv.id,
        "conversation_id": conv.conversation_id,
        "agent_version": conv.agent_version,
        "turns": data.get("turns", []),
        "feedback": conv.feedback_data,
        "metadata": conv.metadata_data,
        "created_at": conv.created_at,
    }


@router.post("", response_model=ConversationResponse, status_code=201)
async def create_conversation(
    payload: ConversationCreate,
    db: AsyncSession = Depends(get_db),
):
    conv_id = payload.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"

    existing = await db.execute(select(Conversation).where(Conversation.conversation_id == conv_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail=f"Conversation '{conv_id}' already exists.")

    turns_data = [t.model_dump(mode="json") for t in payload.turns]
    conv = Conversation(
        conversation_id=conv_id,
        agent_version=payload.agent_version,
        data={"turns": turns_data},
        feedback_data=payload.feedback.model_dump(mode="json") if payload.feedback else None,
        metadata_data=payload.metadata.model_dump(mode="json") if payload.metadata else None,
    )
    db.add(conv)
    await db.commit()
    await db.refresh(conv)
    logger.info("Ingested conversation %s", conv_id)

    try:
        await event_bus.publish(PipelineEvent(
            event_type="conversation_ingested",
            timestamp=datetime.now(timezone.utc),
            data={"conversation_id": conv_id, "turn_count": len(payload.turns)},
            conversation_id=conv_id,
            agent_version=payload.agent_version,
        ).model_dump_json())
    except Exception:
        pass

    return _conv_to_response(conv)


@router.post("/batch", status_code=201)
async def batch_create_conversations(
    payloads: list[ConversationCreate],
    db: AsyncSession = Depends(get_db),
):
    created_ids: list[str] = []
    for payload in payloads:
        conv_id = payload.conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        existing = await db.execute(select(Conversation).where(Conversation.conversation_id == conv_id))
        if existing.scalar_one_or_none():
            logger.warning("Skipping duplicate conversation_id: %s", conv_id)
            continue
        turns_data = [t.model_dump(mode="json") for t in payload.turns]
        conv = Conversation(
            conversation_id=conv_id,
            agent_version=payload.agent_version,
            data={"turns": turns_data},
            feedback_data=payload.feedback.model_dump(mode="json") if payload.feedback else None,
            metadata_data=payload.metadata.model_dump(mode="json") if payload.metadata else None,
        )
        db.add(conv)
        created_ids.append(conv_id)
    await db.commit()
    logger.info("Batch ingested %d conversations", len(created_ids))
    return {"count": len(created_ids), "conversation_ids": created_ids}


@router.get("/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Conversation).where(Conversation.conversation_id == conversation_id))
    conv = result.scalar_one_or_none()
    if not conv:
        raise HTTPException(status_code=404, detail=f"Conversation '{conversation_id}' not found.")
    return _conv_to_response(conv)


@router.get("")
async def list_conversations(
    offset: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    agent_version: str | None = None,
    has_feedback: bool | None = None,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Conversation).offset(offset).limit(limit).order_by(Conversation.created_at.desc())
    if agent_version:
        stmt = stmt.where(Conversation.agent_version == agent_version)
    if has_feedback is True:
        stmt = stmt.where(Conversation.feedback_data.isnot(None))
    elif has_feedback is False:
        stmt = stmt.where(Conversation.feedback_data.is_(None))

    result = await db.execute(stmt)
    convs = result.scalars().all()
    return {
        "data": [_conv_to_response(c) for c in convs],
        "meta": {"count": len(convs), "offset": offset, "limit": limit},
    }
