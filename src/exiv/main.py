import click
import os
import json
import subprocess
import sys
from pathlib import Path

from .utils.logging import app_logger
from .config import global_config
from .components.extension_registry import ExtensionRegistry
from .utils.file import find_file_path, CONFIG_FILENAME

DEFAULT_CONFIG = {"extensions": []}

def _load_config(config_file: Path) -> dict:
    if not config_file.exists():
        return DEFAULT_CONFIG.copy()

    try:
        with config_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data.get("extensions", []), list):
                data["extensions"] = []
            return data
    except (OSError, json.JSONDecodeError) as e:
        app_logger.warning(f"Could not read {config_file}, creating new one. Error: {e}")
        return DEFAULT_CONFIG.copy()

def _save_config(config_file: Path, config: dict) -> None:
    with config_file.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

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
    from .server.server import start_worker
    from .server.task_manager import task_manager

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
    from .server.server import run_server
    app_logger.info("Starting the web server on http://0.0.0.0:8000")
    run_server()

@cli.command(name="register")
@click.argument(
    "path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, resolve_path=True),
)
def register(path: str) -> None:
    """
    - Registers extensions found in a local path (or the path itself if it is an extension)
    - Updates (or creates) the exiv_config.json file with individual extension paths
    - Installs requirements.txt if present for each extension
    - Does NOT imports the extension modules
    """
    config_file_str, config_dir_str = find_file_path(CONFIG_FILENAME, recursive=True)

    if config_file_str:
        config_file = Path(config_file_str)
        config_dir = Path(config_dir_str)
    else:
        config_dir = Path.cwd()
        config_file = config_dir / CONFIG_FILENAME

    config = _load_config(config_file)
    target = Path(path).resolve()
    
    # extensions to add
    extensions_to_add = []
    if (target / "__init__.py").is_file():
        extensions_to_add.append(target)
    else:
        for item in target.iterdir():
            if item.is_dir() and (item / "__init__.py").is_file():
                extensions_to_add.append(item)
    if not extensions_to_add:
        app_logger.warning(f"No valid extensions found in {target} (looking for folders with __init__.py)")
        return

    existing_paths = config.get("extensions", [])
    existing_norm = set()
    for p in existing_paths:
        p_obj = Path(p)
        if not p_obj.is_absolute():
            p_obj = (config_dir / p_obj).resolve()
        existing_norm.add(str(p_obj))

    updated = False
    from .utils.common import install_requirements
    for ext_path in extensions_to_add:
        # 1. Install requirements first
        if not install_requirements(str(ext_path)):
            app_logger.warning(f"Skipping registration of {ext_path.name} due to installation failure.")
            continue

        # 2. Register if successful
        ext_path_resolved = ext_path.resolve()
        if str(ext_path_resolved) not in existing_norm:
            try:
                rel = ext_path_resolved.relative_to(config_dir)
                final_path = str(rel)
            except ValueError:
                final_path = str(ext_path_resolved)

            existing_paths.append(final_path)
            existing_norm.add(str(ext_path_resolved))
            updated = True
            app_logger.info(f"Registered extension: {ext_path.name}")
        else:
            app_logger.info(f"Extension already registered: {ext_path.name}")

    if updated:
        config["extensions"] = existing_paths
        _save_config(config_file, config)
        app_logger.info(f"Updated {config_file}")

@cli.group(name="list")
def list_resources():
    """List resources (extensions, etc.)"""
    # TODO: will extend this for hooks, conditions etc.. later on
    pass

@list_resources.command(name="extensions")
def list_extensions():
    """List all registered extension paths from config"""
    config_file_str, _ = find_file_path(CONFIG_FILENAME, recursive=True)
    current_dir = Path(__file__).resolve().parent
    builtin_dir = current_dir / "extensions"
    print(f"\n[Built-in Extensions Directory]:\n  {builtin_dir}")
    
    if not config_file_str:
        print("\n[Registered Extensions]:\n  (No exiv_config.json found)")
        return

    config = _load_config(Path(config_file_str))
    paths = config.get("extensions", [])
    if not paths:
        print("\n[Registered Extensions]:\n  (None)")
    else:
        print(f"\n[Registered Extensions] (from {config_file_str}):")
        for p in paths:
            print(f"  - {p}")
    print("")

if __name__ == '__main__':
    cli()