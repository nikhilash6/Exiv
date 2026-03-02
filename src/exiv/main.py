import click
import os
import json
import subprocess
import sys
from pathlib import Path

from .utils.logging import app_logger
from .config import global_config
from .components.extension_registry import ExtensionRegistry, EXTENSION_ENTRYPOINT
from .utils.file import find_file_path, CONFIG_FILENAME


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
    from .server.server import run_server, load_server_config
    load_server_config()
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
    - Updates (or creates) the config.json file with individual extension paths
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

    config = ExtensionRegistry.load_config(config_file)
    target = Path(path).resolve()
    
    # extensions to add
    extensions_to_add = []
    if (target / EXTENSION_ENTRYPOINT).is_file():
        extensions_to_add.append(target)
    else:
        for item in target.iterdir():
            if item.is_dir() and (item / EXTENSION_ENTRYPOINT).is_file():
                extensions_to_add.append(item)
    if not extensions_to_add:
        app_logger.warning(f"No valid extensions found in {target} (looking for folders with {EXTENSION_ENTRYPOINT})")
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
        ExtensionRegistry.save_config(config_file, existing_paths)
        app_logger.info(f"Updated {config_file}")

@cli.group(name="list")
def list_resources():
    """List resources (extensions, etc.)"""
    # TODO: will extend this for hooks, conditions etc.. later on
    pass

@list_resources.command(name="extensions")
def list_extensions():
    """List all registered extension paths from config with metadata"""
    registry = ExtensionRegistry.get_instance()
    if not (extensions_metadata:=registry.get_all_extensions_metadata()):
        print("No extensions registered.")
        return
    
    # column widths
    headers = ["ID", "Name", "Version", "Path"]
    widths = [len(h) for h in headers]
    for meta in extensions_metadata:
        widths[0] = max(widths[0], len(str(meta.get('id', 'N/A'))))
        widths[1] = max(widths[1], len(str(meta.get('name', 'N/A'))))
        widths[2] = max(widths[2], len(str(meta.get('version', 'N/A'))))
        widths[3] = max(widths[3], len(str(meta.get('path', 'N/A'))))

    widths = [w + 2 for w in widths]    # padding
    header_str = "".join(h.ljust(w) for h, w in zip(headers, widths))
    print(f"\n{header_str}")
    print("-" * sum(widths))
    for meta in extensions_metadata:
        row = [
            str(meta.get('id', 'N/A')),
            str(meta.get('name', 'N/A')),
            str(meta.get('version', 'N/A')),
            str(meta.get('path', 'N/A'))
        ]
        print("".join(val.ljust(w) for val, w in zip(row, widths)))
    print("")

if __name__ == '__main__':
    cli()