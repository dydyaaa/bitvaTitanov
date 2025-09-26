import pytest


@pytest.mark.anyio
async def test_register_and_login(client):
    r = await client.post("/users/register", json={"username": "u1", "password": "p1"})
    assert r.status_code == 200
    assert r.json()["username"] == "u1"

    r = await client.post("/users/login", json={"username": "u1", "password": "p1"})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
