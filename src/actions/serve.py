from src.core.serve.server import Server
import uvicorn


def run_server(model_dir, model_tag, host, port):
    server = Server(model_dir=model_dir, model_tag=model_tag)
    uvicorn.run(server.app, host=host, port=port)
