import click
import os
import json
import subprocess
import sys

from .server.server import run_server, start_worker
from .server.task_manager import task_manager
from .utils.logging import app_logger
from .config import global_config
from .components.extension_registry import ExtensionRegistry

@click.group()
def cli():
    pass

@cli.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument('app_name')
@click.pass_context
def run(ctx, app_name):
    """ Runs a task synchronously """
    metadata = {}
    # ctx.args will be a list like ['--seed', '12345', '--negative-prompt', 'blurry']
    for i, arg in enumerate(ctx.args):
        if arg.startswith('--'):
            key = arg[2:]
            if i + 1 < len(ctx.args) and not ctx.args[i+1].startswith('--'):
                metadata[key] = ctx.args[i+1]
            else:
                # treat it as a flag (e.g., --low_vram)
                metadata[key] = True
                
    global_config.update_config(metadata)
    app_logger.set_level(global_config.logging_level)

    task_id = task_manager.add_task(app_name=app_name, params=metadata)
    start_worker(sync_mode=True)
    return task_manager.get_task_progress(task_id=task_id)


@cli.command(name="serve")
def serve():
    app_logger.info("Starting the web server on http://0.0.0.0:8000")
    run_server()

@cli.command(name="register")
@click.argument('path', type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True))
def register(path):
    """
    - Registers a local folder as an extensions folder
    - Updates (or creates) the .exivrc file in the current directory
    - Installs requirements.txt if present
    - Does NOT imports the extension modules (see load_extensions_from_path)
    """
    app_logger.info(f"Registering extension from: {path}")
    
    # update .exivrc
    config_path = os.path.join(os.getcwd(), ".exivrc")
    config = {"extensions": []}
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except Exception as e:
            app_logger.warning(f"Could not read existing .exivrc, creating new one. Error: {e}")

    # using relative path if possible for portability, else absolute
    try:
        rel_path = os.path.relpath(path, os.getcwd())
        final_path = rel_path
    except ValueError:
        final_path = path
        
    if final_path not in config.get("extensions", []):
        config["extensions"] = config.get("extensions", []) + [final_path]
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        app_logger.info(f"Added {final_path} to .exivrc")
    else:
        app_logger.info(f"Extension {final_path} is already registered.")

    # dependencies
    req_file = os.path.join(path, "requirements.txt")
    if os.path.exists(req_file):
        app_logger.info(f"Found requirements.txt, installing dependencies...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
            app_logger.info("Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            app_logger.error(f"Failed to install dependencies: {e}")

@cli.group(name="list")
def list_resources():
    """List resources (extensions, etc.)"""
    # TODO: will extend this for hooks, conditions etc.. later on
    pass

@list_resources.command(name="extensions")
def list_extensions():
    """List all available extensions (built-in + registered)"""
    registry = ExtensionRegistry.get_instance()
    registry.initialize()
    
    meta = registry.get_all_extensions_metadata()
    if not meta:
        print("No extensions found.")
        return

    print(f"\nFound {len(meta)} extensions:\n")
    print(f"{'ID':<30} {'Version':<10} {'Name':<20} {'Slot'}")
    print("-" * 70)
    for ext in meta:
        print(f"{ext['id']:<30} {ext['version']:<10} {ext['name']:<20} {ext['slot']}")
    print("")

if __name__ == '__main__':
    cli()