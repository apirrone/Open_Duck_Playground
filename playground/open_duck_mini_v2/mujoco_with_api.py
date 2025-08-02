#!/usr/bin/env python3
"""
Launcher script to run MuJoCo simulation with API server
"""
import argparse
import threading
import uvicorn
import sys
import time
from contextlib import asynccontextmanager

from playground.open_duck_mini_v2.mujoco_infer import MjInfer
from playground.open_duck_mini_v2.api_server import app

# Global reference to MjInfer instance
mjinfer_instance = None

@asynccontextmanager
async def lifespan(app):
    # Startup
    print("API server started. MuJoCo simulation is running...")
    yield
    # Shutdown
    print("Shutting down API server...")

# Update the app lifespan
app.router.lifespan_context = lifespan

def run_api_server(host="0.0.0.0", port=8000):
    """Run the FastAPI server in a separate thread"""
    config = uvicorn.Config(
        app, 
        host=host, 
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    server.run()

def main():
    parser = argparse.ArgumentParser(description="Run Open Duck MuJoCo simulation with API server")
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--reference_data",
        type=str,
        default="playground/open_duck_mini_v2/data/polynomial_coefficients.pkl",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml",
    )
    parser.add_argument("--standing", action="store_true", default=False)
    parser.add_argument("--api_host", type=str, default="0.0.0.0", help="API server host")
    parser.add_argument("--api_port", type=int, default=8000, help="API server port")
    
    args = parser.parse_args()
    
    # Start API server in a background thread
    api_thread = threading.Thread(
        target=run_api_server,
        args=(args.api_host, args.api_port),
        daemon=True
    )
    api_thread.start()
    
    # Give the API server time to start
    print(f"Starting API server on {args.api_host}:{args.api_port}...")
    time.sleep(2)
    
    # Create and run MuJoCo simulation with API mode enabled
    global mjinfer_instance
    mjinfer_instance = MjInfer(
        args.model_path, 
        args.reference_data, 
        args.onnx_model_path, 
        args.standing,
        use_api=True  # Always use API mode when launched this way
    )
    
    print("Starting MuJoCo simulation with API control enabled...")
    print(f"API endpoints available at http://{args.api_host}:{args.api_port}")
    print("Press 'h' in the viewer to toggle between locomotion/head control modes")
    
    try:
        mjinfer_instance.run()
    except KeyboardInterrupt:
        print("\nShutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()