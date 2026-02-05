from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from oghma.config import (
    create_default_config,
    get_config_path,
    load_config,
    validate_config,
)
from oghma.storage import Storage

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="oghma")
def cli() -> None:
    pass


@cli.command()
def init() -> None:
    config_path = get_config_path()

    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not click.confirm("Overwrite existing config?"):
            console.print("[green]Init cancelled[/green]")
            return

    console.print("[blue]Creating Oghma configuration...[/blue]")
    config = create_default_config()
    console.print(f"[green]Config created at {config_path}[/green]")
    console.print(f"[cyan]Database path: {config['storage']['db_path']}[/cyan]")
    console.print("\n[yellow]Run 'oghma status' to verify setup[/yellow]")


@cli.command()
def status() -> None:
    try:
        config_path = get_config_path()
        config = load_config()
        db_path = config["storage"]["db_path"]

        table = Table(title="Oghma Status", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Config Path", str(config_path))
        table.add_row("Database Path", db_path)

        if Path(db_path).exists():
            storage = Storage(db_path, config)
            memory_count = storage.get_memory_count()
            table.add_row("Memory Count", str(memory_count))
            table.add_row("Database Status", "[green]Exists[/green]")
        else:
            table.add_row("Memory Count", "0")
            table.add_row("Database Status", "[yellow]Not created yet[/yellow]")

        console.print(table)

        errors = validate_config(config)
        if errors:
            console.print("\n[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  [red]- {error}[/red]")

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


def main() -> None:
    cli()
