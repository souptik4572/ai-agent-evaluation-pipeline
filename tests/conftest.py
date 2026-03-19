import asyncio
from datetime import datetime, timezone
from typing import AsyncGenerator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.database import Base, get_db
from app.main import app
from app.models.schemas import (
    Annotation,
    ConversationCreate,
    ConversationMetadata,
    Feedback,
    OpsReview,
    ToolCall,
    Turn,
)

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture(scope="session")
async def test_engine():
    engine = create_async_engine(TEST_DB_URL, echo=False)
    async with engine.begin() as conn:
        from app.models import db_models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)
    async with session_factory() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def client(test_engine) -> AsyncGenerator[AsyncClient, None]:
    session_factory = async_sessionmaker(test_engine, expire_on_commit=False, class_=AsyncSession)

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


def _ts(offset: int = 0) -> datetime:
    from datetime import timedelta
    return datetime.now(timezone.utc) + timedelta(seconds=offset)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_conversation() -> ConversationCreate:
    return ConversationCreate(
        conversation_id="conv_test_001",
        agent_version="v2.3.1",
        turns=[
            Turn(turn_id=1, role="user", content="Book a flight NYC to London next week.", timestamp=_ts(0)),
            Turn(
                turn_id=2,
                role="assistant",
                content="Searching for flights for you.",
                tool_calls=[
                    ToolCall(
                        tool_name="flight_search",
                        parameters={"origin": "NYC", "destination": "LHR", "date_range": "2024-01-22/2024-01-29"},
                        result={"status": "success", "flights": [{"id": "BA112", "price": 650}]},
                        latency_ms=320,
                    )
                ],
                timestamp=_ts(5),
            ),
            Turn(turn_id=3, role="user", content="Book the cheapest one.", timestamp=_ts(10)),
            Turn(turn_id=4, role="assistant", content="Booked BA112 for $650. Confirmation: XK-4821.", timestamp=_ts(15)),
        ],
        feedback=Feedback(
            user_rating=4,
            ops_review=OpsReview(quality="good"),
            annotations=[Annotation(type="tool_accuracy", label="correct", annotator_id="ann_001", confidence=0.9)],
        ),
        metadata=ConversationMetadata(total_latency_ms=800, mission_completed=True),
    )


@pytest.fixture
def sample_conversation_with_tool_error() -> ConversationCreate:
    return ConversationCreate(
        conversation_id="conv_test_tool_err",
        agent_version="v2.3.1",
        turns=[
            Turn(turn_id=1, role="user", content="Find flights to London next week.", timestamp=_ts(0)),
            Turn(
                turn_id=2,
                role="assistant",
                content="Searching...",
                tool_calls=[
                    ToolCall(
                        tool_name="flight_search",
                        parameters={"origin": "NYC", "destination": "LHR", "date_range": "next week"},
                        result={"status": "error", "message": "Invalid date format"},
                        latency_ms=1200,
                    )
                ],
                timestamp=_ts(5),
            ),
        ],
        feedback=Feedback(user_rating=2),
        metadata=ConversationMetadata(total_latency_ms=1200, mission_completed=False),
    )


@pytest.fixture
def sample_long_conversation() -> ConversationCreate:
    pref_statement = "I prefer window seats and my budget is under $500."
    turns = [
        Turn(turn_id=1, role="user", content=f"Hi I need flights to Miami. {pref_statement}", timestamp=_ts(0)),
        Turn(turn_id=2, role="assistant", content="Got it! I'll keep your preferences in mind.", timestamp=_ts(5)),
        Turn(turn_id=3, role="user", content="What options do you have?", timestamp=_ts(10)),
        Turn(turn_id=4, role="assistant", content="Let me look that up.", timestamp=_ts(15)),
        Turn(turn_id=5, role="user", content="What about direct flights?", timestamp=_ts(20)),
        Turn(turn_id=6, role="assistant", content="Here is a flight for $620 with a middle seat departing at 9am.", timestamp=_ts(25)),
    ]
    return ConversationCreate(
        conversation_id="conv_test_long",
        agent_version="v2.3.1",
        turns=turns,
        feedback=Feedback(user_rating=1),
        metadata=ConversationMetadata(total_latency_ms=1400, mission_completed=False),
    )


@pytest.fixture
def sample_conversation_with_disagreement() -> ConversationCreate:
    return ConversationCreate(
        conversation_id="conv_test_disagree",
        agent_version="v2.3.1",
        turns=[
            Turn(turn_id=1, role="user", content="Can you help me plan a trip?", timestamp=_ts(0)),
            Turn(turn_id=2, role="assistant", content="Sure! Where would you like to go?", timestamp=_ts(3)),
        ],
        feedback=Feedback(
            user_rating=3,
            annotations=[
                Annotation(type="response_quality", label="good", annotator_id="ann_201", confidence=0.75),
                Annotation(type="response_quality", label="good", annotator_id="ann_202", confidence=0.65),
                Annotation(type="response_quality", label="poor", annotator_id="ann_203", confidence=0.80),
            ],
        ),
        metadata=ConversationMetadata(total_latency_ms=600, mission_completed=True),
    )
