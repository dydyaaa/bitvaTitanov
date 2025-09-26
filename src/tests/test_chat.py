import pytest


@pytest.mark.anyio
async def test_chat_flow(client, auth_headers):
    # # Отправка сообщения
    # r = await client.post("/chat/send", json={"text": "Привет бот!"}, headers=auth_headers)
    # assert r.status_code == 200
    # assert r.json()["response"] == "Привет"

    # Получение истории
    r = await client.get("/chat/history", headers=auth_headers)
    assert r.status_code == 200
    history = r.json()
    assert len(history) == 1

    # Очистка истории
    r = await client.delete("/chat/history", headers=auth_headers)
    assert r.status_code == 200
    assert r.json()["status"] == "cleared"

    # Проверка, что история пустая
    r = await client.get("/chat/history", headers=auth_headers)
    assert r.status_code == 200
    assert r.json() == []
