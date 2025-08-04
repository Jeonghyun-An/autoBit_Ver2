# app/routers/ws.py
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.ws.manager import ws_manager

router = APIRouter()

@router.websocket("/ws/trades")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # (옵션: ping/pong 같은 것)
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
