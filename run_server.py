"""
run_server.py - Local development server launcher with extended WebSocket timeout.
Run this instead of uvicorn directly:
    python run_server.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        # Disable WebSocket keepalive pings to prevent disconnects during inference
        ws_ping_interval=None,
    )
