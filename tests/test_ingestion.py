import pytest


@pytest.mark.asyncio
async def test_post_conversation_valid(client, sample_conversation):
    payload = sample_conversation.model_dump(mode="json")
    resp = await client.post("/api/v1/conversations", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["conversation_id"] == "conv_test_001"
    assert data["agent_version"] == "v2.3.1"
    assert len(data["turns"]) == 4


@pytest.mark.asyncio
async def test_post_conversation_missing_required_fields(client):
    resp = await client.post("/api/v1/conversations", json={"agent_version": "v1.0"})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_get_conversation(client, sample_conversation):
    payload = sample_conversation.model_dump(mode="json")
    payload["conversation_id"] = "conv_get_test"
    await client.post("/api/v1/conversations", json=payload)

    resp = await client.get("/api/v1/conversations/conv_get_test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["conversation_id"] == "conv_get_test"


@pytest.mark.asyncio
async def test_get_conversation_not_found(client):
    resp = await client.get("/api/v1/conversations/does_not_exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_batch_ingestion(client, sample_conversation):
    items = []
    for i in range(3):
        p = sample_conversation.model_dump(mode="json")
        p["conversation_id"] = f"conv_batch_{i}"
        items.append(p)
    resp = await client.post("/api/v1/conversations/batch", json=items)
    assert resp.status_code == 201
    data = resp.json()
    assert data["count"] == 3
    assert len(data["conversation_ids"]) == 3


@pytest.mark.asyncio
async def test_list_conversations(client, sample_conversation):
    payload = sample_conversation.model_dump(mode="json")
    payload["conversation_id"] = "conv_list_test"
    await client.post("/api/v1/conversations", json=payload)

    resp = await client.get("/api/v1/conversations?limit=10")
    assert resp.status_code == 200
    data = resp.json()
    assert "data" in data
    assert "meta" in data
