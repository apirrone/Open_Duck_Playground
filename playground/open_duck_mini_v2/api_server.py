import asyncio
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import json
import uvicorn

from playground.open_duck_mini_v2.command_handler import CommandHandler, ControlMode

app = FastAPI(title="Open Duck Robot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global command handler instance
command_handler = CommandHandler()

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Pydantic models
class LocomotionCommand(BaseModel):
    forward: float = Field(default=0.0, ge=-0.15, le=0.15)
    lateral: float = Field(default=0.0, ge=-0.2, le=0.2)
    angular: float = Field(default=0.0, ge=-1.0, le=1.0)

class HeadCommand(BaseModel):
    neck_pitch: float = Field(default=0.0, ge=-0.34, le=1.1)
    head_pitch: float = Field(default=0.0, ge=-0.78, le=0.78)
    head_yaw: float = Field(default=0.0, ge=-1.5, le=1.5)
    head_roll: float = Field(default=0.0, ge=-0.5, le=0.5)

class MoveCommand(BaseModel):
    direction: str = Field(..., pattern="^(forward|backward|left|right)$")
    speed: Optional[float] = None

class TurnCommand(BaseModel):
    direction: str = Field(..., pattern="^(left|right)$")
    speed: Optional[float] = None

class ControlModeRequest(BaseModel):
    mode: str = Field(..., pattern="^(locomotion|head)$")

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Open Duck Robot API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/status")
async def get_status():
    return command_handler.get_status()

@app.post("/command/locomotion")
async def set_locomotion_command(cmd: LocomotionCommand):
    command_handler.set_locomotion_command(cmd.forward, cmd.lateral, cmd.angular)
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": "Locomotion command set", "commands": status["commands"]}

@app.post("/command/head")
async def set_head_command(cmd: HeadCommand):
    command_handler.set_head_command(
        cmd.neck_pitch, cmd.head_pitch, cmd.head_yaw, cmd.head_roll
    )
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": "Head command set", "commands": status["commands"]}

@app.post("/command/move/{direction}")
async def move(direction: str, speed: Optional[float] = None):
    if direction == "forward":
        command_handler.move_forward(speed)
    elif direction == "backward":
        command_handler.move_backward(speed)
    elif direction == "left":
        command_handler.move_left(speed)
    elif direction == "right":
        command_handler.move_right(speed)
    else:
        raise HTTPException(status_code=400, detail="Invalid direction")
    
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": f"Moving {direction}", "commands": status["commands"]}

@app.post("/command/turn/{direction}")
async def turn(direction: str, speed: Optional[float] = None):
    if direction == "left":
        command_handler.turn_left(speed)
    elif direction == "right":
        command_handler.turn_right(speed)
    else:
        raise HTTPException(status_code=400, detail="Invalid direction")
    
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": f"Turning {direction}", "commands": status["commands"]}

@app.post("/command/stop")
async def stop():
    command_handler.stop()
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": "Robot stopped", "commands": status["commands"]}

@app.post("/command/emergency_stop")
async def emergency_stop():
    command_handler.emergency_stop()
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": "Emergency stop activated", "commands": status["commands"]}

@app.post("/command/reset_head")
async def reset_head():
    command_handler.reset_head()
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": "Head reset to neutral position", "commands": status["commands"]}

@app.post("/control_mode")
async def set_control_mode(request: ControlModeRequest):
    mode = ControlMode.LOCOMOTION if request.mode == "locomotion" else ControlMode.HEAD
    command_handler.set_control_mode(mode)
    status = command_handler.get_status()
    await manager.broadcast({"type": "status", "data": status})
    return {"message": f"Control mode set to {request.mode}", "mode": request.mode}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status
        await websocket.send_json({
            "type": "connected",
            "data": command_handler.get_status()
        })
        
        while True:
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get("type") == "locomotion":
                cmd = data.get("data", {})
                command_handler.set_locomotion_command(
                    cmd.get("forward", 0.0),
                    cmd.get("lateral", 0.0),
                    cmd.get("angular", 0.0)
                )
            elif data.get("type") == "head":
                cmd = data.get("data", {})
                command_handler.set_head_command(
                    cmd.get("neck_pitch", 0.0),
                    cmd.get("head_pitch", 0.0),
                    cmd.get("head_yaw", 0.0),
                    cmd.get("head_roll", 0.0)
                )
            elif data.get("type") == "stop":
                command_handler.stop()
            elif data.get("type") == "emergency_stop":
                command_handler.emergency_stop()
            elif data.get("type") == "control_mode":
                mode = data.get("mode", "locomotion")
                mode_enum = ControlMode.LOCOMOTION if mode == "locomotion" else ControlMode.HEAD
                command_handler.set_control_mode(mode_enum)
            
            # Broadcast status to all clients
            status = command_handler.get_status()
            await manager.broadcast({"type": "status", "data": status})
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def get_command_handler():
    """Get the global command handler instance for integration with mujoco_infer"""
    return command_handler

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)