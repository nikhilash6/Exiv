import click

from .server.server import run_server, start_worker
from .server.task_manager import task_manager
from .utils.logging import app_logger
from .config import global_config

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

if __name__ == '__main__':
    cli()