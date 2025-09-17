import click
from . import server

@click.group()
def cli():
    pass

@cli.command()
@click.argument('task_name')
@click.option('--prompt', required=True, help='The prompt for the task.')
@click.option('--negative-prompt', help='The negative prompt for the task.')
@click.option('--seed', type=int, help='The seed for the task.')
def run(task_name, prompt, negative_prompt, seed):
    """Runs a task synchronously."""
    print(f"Running task: {task_name}")
    print(f"  Prompt: {prompt}")
    if negative_prompt:
        print(f"  Negative Prompt: {negative_prompt}")
    if seed:
        print(f"  Seed: {seed}")
    # Simulate work and return result
    print("\nResult: A photograph of a golden retriever at the park")

class Task:
    def __init__(self, task_type, task_name):
        self.task_type = task_type
        self.task_name = task_name

    def __call__(self, **kwargs):
        print(f"Running task: {self.task_type}:{self.task_name}")
        # Simulate work
        return "A photograph of a golden retriever at the park"

    @staticmethod
    def get(task_type, task_name):
        return Task(task_type, task_name)

@cli.command()
def serve():
    """Starts the web server."""
    print("Starting the web server on http://0.0.0.0:8000")
    server.run_server()

if __name__ == '__main__':
    cli()