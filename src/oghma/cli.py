import json
import os
import signal
import time
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from oghma import __version__
from oghma.config import (
    create_default_config,
    get_config_path,
    load_config,
    validate_config,
)
from oghma.daemon import Daemon, get_daemon_pid
from oghma.embedder import EmbedConfig, create_embedder
from oghma.exporter import Exporter, ExportOptions
from oghma.migration import EmbeddingMigration
from oghma.storage import Storage

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="oghma")
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
@click.option("--json", "as_json", is_flag=True, help="Output status as JSON")
def status(as_json: bool) -> None:
    try:
        config_path = get_config_path()
        config = load_config()
        db_path = config["storage"]["db_path"]
        pid_file = config["daemon"]["pid_file"]

        status_payload = {
            "config_path": str(config_path),
            "daemon": {"running": False, "pid": None},
            "database": {"path": db_path, "exists": False},
            "memory_count": 0,
            "last_extraction": None,
            "watched_files": 0,
            "config_errors": [],
        }

        table = Table(title="Oghma Status", show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Config Path", str(config_path))

        pid = get_daemon_pid(pid_file)
        if pid:
            status_payload["daemon"] = {"running": True, "pid": pid}
            table.add_row("Daemon Status", f"[green]Running (PID: {pid})[/green]")
        else:
            table.add_row("Daemon Status", "[red]Stopped[/red]")

        table.add_row("Database Path", db_path)

        if Path(db_path).exists():
            status_payload["database"]["exists"] = True
            storage = Storage(db_path, config)
            memory_count = storage.get_memory_count()
            status_payload["memory_count"] = memory_count
            table.add_row("Memory Count", str(memory_count))

            logs = storage.get_recent_extraction_logs(limit=1)
            if logs:
                last_extraction = logs[0]["created_at"]
                status_payload["last_extraction"] = last_extraction
                table.add_row("Last Extraction", last_extraction)
            else:
                table.add_row("Last Extraction", "Never")

            table.add_row("Database Status", "[green]Exists[/green]")

            from oghma.watcher import Watcher

            watcher = Watcher(config, storage)
            watched_files = watcher.discover_files()
            status_payload["watched_files"] = len(watched_files)
            table.add_row("Watched Files", str(len(watched_files)))
        else:
            table.add_row("Memory Count", "0")
            table.add_row("Last Extraction", "Never")
            table.add_row("Database Status", "[yellow]Not created yet[/yellow]")
            table.add_row("Watched Files", "0")

        errors = validate_config(config)
        if errors:
            status_payload["config_errors"] = errors

        if as_json:
            console.print(json.dumps(status_payload, indent=2))
            return

        console.print(table)
        if errors:
            console.print("\n[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  [red]- {error}[/red]")

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@cli.command("validate-config")
def validate_config_command() -> None:
    try:
        config = load_config()
        errors = validate_config(config)
        if errors:
            console.print("[red]Configuration errors:[/red]")
            for error in errors:
                console.print(f"  [red]- {error}[/red]")
            raise SystemExit(1)

        console.print("[green]Configuration OK[/green]")
    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
@click.option("--foreground", "-f", is_flag=True, help="Run in foreground (don't daemonize)")
def start(foreground: bool) -> None:
    try:
        config = load_config()
        pid_file = config["daemon"]["pid_file"]

        pid = get_daemon_pid(pid_file)
        if pid:
            console.print(f"[red]Daemon already running (PID: {pid})[/red]")
            console.print("Use 'oghma stop' to stop it first.")
            raise SystemExit(1)

        console.print("[blue]Starting Oghma daemon...[/blue]")

        if not foreground:
            try:
                pid = os.fork()
                if pid > 0:
                    console.print(f"[green]Daemon started in background (PID: {pid})[/green]")
                    return
            except OSError as e:
                console.print(f"[yellow]Fork failed: {e}. Running in foreground.[/yellow]")

        daemon = Daemon(config)
        daemon.start()

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error starting daemon: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
def stop() -> None:
    try:
        config = load_config()
        pid_file = config["daemon"]["pid_file"]

        pid = get_daemon_pid(pid_file)
        if not pid:
            console.print("[yellow]Daemon is not running[/yellow]")
            return

        console.print(f"[blue]Stopping daemon (PID: {pid})...[/blue]")

        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            console.print("[yellow]Daemon process not found. Cleaning up PID file.[/yellow]")
            Path(pid_file).unlink(missing_ok=True)
            return

        for _ in range(10):
            time.sleep(0.5)
            if not get_daemon_pid(pid_file):
                console.print("[green]Daemon stopped successfully[/green]")
                return

        console.print("[yellow]Daemon did not stop gracefully. Sending SIGKILL...[/yellow]")
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

        Path(pid_file).unlink(missing_ok=True)
        console.print("[green]Daemon force stopped[/green]")

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error stopping daemon: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
@click.option(
    "--status",
    type=click.Choice(["active", "archived"]),
    default="active",
    show_default=True,
    help="Filter by status",
)
def stats(status: str) -> None:
    """Show memory stats by category and source tool."""
    try:
        config = load_config()
        storage = Storage(config=config)
        memories = storage.get_all_memories(status=status)

        if not memories:
            console.print(f"[yellow]No {status} memories found[/yellow]")
            return

        category_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        for memory in memories:
            category_counts[memory["category"]] = category_counts.get(memory["category"], 0) + 1
            source_counts[memory["source_tool"]] = source_counts.get(memory["source_tool"], 0) + 1

        console.print(f"[cyan]Total {status} memories:[/cyan] {len(memories)}\n")

        category_table = Table(title="By Category", show_header=True, header_style="bold magenta")
        category_table.add_column("Category", style="cyan")
        category_table.add_column("Count", style="green")
        for category, count in sorted(category_counts.items()):
            category_table.add_row(category, str(count))
        console.print(category_table)
        console.print()

        source_table = Table(title="By Source Tool", show_header=True, header_style="bold magenta")
        source_table.add_column("Source Tool", style="cyan")
        source_table.add_column("Count", style="green")
        for source, count in sorted(source_counts.items()):
            source_table.add_row(source, str(count))
        console.print(source_table)

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
@click.option(
    "--threshold",
    "-t",
    default=0.92,
    show_default=True,
    help="Cosine similarity threshold (0.0-1.0)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=True,
    show_default=True,
    help="Preview without deleting",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually delete duplicates (overrides --dry-run)",
)
@click.option("--category", "-c", help="Only dedup within this category")
@click.option("--batch-size", default=500, show_default=True, help="Processing batch size")
def dedup(
    threshold: float,
    dry_run: bool,
    execute: bool,
    category: str | None,
    batch_size: int,
) -> None:
    """Find and remove semantically duplicate memories."""
    try:
        config = load_config()
        storage = Storage(config=config)

        if execute:
            dry_run = False

        mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE[/red]"
        console.print(f"[blue]Running semantic dedup ({mode}, threshold={threshold})...[/blue]")

        from oghma.dedup import run_dedup

        result = run_dedup(
            storage,
            threshold=threshold,
            category=category,
            dry_run=dry_run,
            batch_size=batch_size,
        )

        table = Table(title="Dedup Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total memories", str(result.total_memories))
        table.add_row("Memories with embeddings", str(result.embedded_memories))
        table.add_row("Duplicate clusters found", str(result.clusters_found))
        table.add_row("Duplicates to remove", str(result.duplicates_removed))
        table.add_row(
            "Memories kept",
            str(result.embedded_memories - result.duplicates_removed),
        )
        console.print(table)

        if dry_run and result.duplicates_removed > 0:
            console.print(
                "\n[yellow]This was a dry run. "
                "Use --execute to actually delete duplicates.[/yellow]"
            )
        elif not dry_run and result.duplicates_removed > 0:
            console.print(
                f"\n[green]Deleted {result.duplicates_removed} duplicate memories.[/green]"
            )
        else:
            console.print("\n[green]No duplicates found above threshold.[/green]")

    except ImportError as e:
        console.print(f"[red]{e}[/red]")
        raise SystemExit(1) from None
    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error during dedup: {e}[/red]")
        raise SystemExit(1) from None


@cli.command("purge-noise")
@click.option(
    "--dry-run",
    is_flag=True,
    default=True,
    show_default=True,
    help="Preview without deleting",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually delete noisy memories (overrides --dry-run)",
)
def purge_noise(dry_run: bool, execute: bool) -> None:
    """Remove memories matching known noise patterns."""
    try:
        config = load_config()
        storage = Storage(config=config)

        if execute:
            dry_run = False

        mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE[/red]"
        console.print(f"[blue]Running noise purge ({mode})...[/blue]")

        from oghma.dedup import run_purge

        result = run_purge(storage, dry_run=dry_run)

        table = Table(title="Noise Purge Results", show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total memories", str(result.total_memories))
        table.add_row("Noise matches", str(result.noise_found))
        table.add_row("Would keep", str(result.total_memories - result.noise_found))
        console.print(table)

        if result.by_reason:
            reason_table = Table(
                title="Matches by Pattern", show_header=True, header_style="bold magenta"
            )
            reason_table.add_column("Pattern", style="cyan")
            reason_table.add_column("Count", style="green")
            for reason, count in sorted(result.by_reason.items(), key=lambda x: -x[1]):
                reason_table.add_row(reason, str(count))
            console.print(reason_table)

        if dry_run and result.noise_found > 0:
            console.print(
                "\n[yellow]This was a dry run. "
                "Use --execute to actually delete noisy memories.[/yellow]"
            )
        elif not dry_run and result.noise_found > 0:
            console.print(f"\n[green]Purged {result.noise_found} noisy memories.[/green]")
        else:
            console.print("\n[green]No noisy memories found.[/green]")

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error during purge: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
@click.argument("query")
@click.option("--limit", "-n", default=10, help="Max results")
@click.option("--category", "-c", help="Filter by category")
@click.option("--tool", "-t", help="Filter by source tool")
@click.option(
    "--status",
    type=click.Choice(["active", "archived"]),
    default="active",
    show_default=True,
    help="Filter by status",
)
@click.option(
    "--mode",
    type=click.Choice(["keyword", "vector", "hybrid"]),
    default="keyword",
    show_default=True,
    help="Search strategy",
)
def search(
    query: str,
    limit: int,
    category: str | None,
    tool: str | None,
    status: str,
    mode: str,
) -> None:
    try:
        config = load_config()
        storage = Storage(config=config)
        query_embedding: list[float] | None = None

        if mode in {"vector", "hybrid"}:
            embed_config = config.get("embedding", {})
            embedder = create_embedder(EmbedConfig.from_dict(embed_config))
            query_embedding = embedder.embed(query)

        results = storage.search_memories_hybrid(
            query=query,
            query_embedding=query_embedding,
            limit=limit,
            category=category,
            source_tool=tool,
            status=status,
            search_mode=mode,
        )

        if not results:
            console.print(f"[yellow]No memories found matching: {query}[/yellow]")
            return

        console.print(f"[cyan]Found {len(results)} memories matching: {query}[/cyan]\n")

        for idx, memory in enumerate(results, 1):
            table = Table(show_header=False, box=None, padding=(0, 0))
            table.add_column("", style="cyan")
            table.add_column("")

            table.add_row(f"[bold]#{idx}[/bold]", f"[dim]{memory['created_at']}[/dim]")
            table.add_row("Category", f"[green]{memory['category']}[/green]")
            table.add_row("Source", f"{memory['source_tool']} ({Path(memory['source_file']).name})")
            table.add_row("Confidence", f"{memory['confidence']:.0%}")
            table.add_row("Content", memory["content"])

            console.print(table)
            console.print()

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error searching memories: {e}[/red]")
        raise SystemExit(1) from None


@cli.command("migrate-embeddings")
@click.option("--batch-size", default=100, show_default=True, help="Batch size")
@click.option("--dry-run", is_flag=True, help="Preview migration without writing embeddings")
def migrate_embeddings(batch_size: int, dry_run: bool) -> None:
    try:
        config = load_config()
        storage = Storage(config=config)

        done_before, total = storage.get_embedding_progress()
        console.print(
            f"[blue]Embedding progress before migration:[/blue] {done_before}/{total} memories"
        )

        if done_before == total and total > 0:
            console.print("[green]All active memories already have embeddings.[/green]")
            return

        embed_config = config.get("embedding", {})
        embedder = create_embedder(EmbedConfig.from_dict(embed_config, batch_size=batch_size))

        migration = EmbeddingMigration(
            storage=storage,
            embedder=embedder,
            batch_size=batch_size,
        )
        result = migration.run(dry_run=dry_run)

        done_after, total_after = storage.get_embedding_progress()
        if dry_run:
            console.print(
                f"[yellow]Dry run complete.[/yellow] Would process {result.processed} memories."
            )
            return

        console.print(
            "[green]Migration complete.[/green] "
            f"Processed={result.processed}, migrated={result.migrated}, "
            f"failed={result.failed}, skipped={result.skipped}"
        )
        console.print(
            f"[cyan]Embedding progress after migration:[/cyan] {done_after}/{total_after} memories"
        )
    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error migrating embeddings: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output directory")
@click.option("--format", "-f", type=click.Choice(["markdown", "json"]), default="markdown")
@click.option(
    "--group-by", "-g", type=click.Choice(["category", "date", "source"]), default="category"
)
@click.option("--category", "-c", help="Export only this category")
@click.option(
    "--status",
    type=click.Choice(["active", "archived"]),
    default="active",
    show_default=True,
    help="Export by status",
)
@click.option("--source-tool", "-t", help="Export only this source tool")
def export(
    output: str | None,
    format: str,
    group_by: str,
    category: str | None,
    status: str,
    source_tool: str | None,
) -> None:
    """Export memories to files."""
    try:
        config = load_config()
        storage = Storage(config=config)

        output_dir = Path(output or config["export"]["output_dir"])

        options = ExportOptions(
            output_dir=output_dir,
            format=format,
            group_by=group_by,
            status=status,
            source_tool=source_tool,
        )
        exporter = Exporter(storage, options)

        if category:
            console.print(f"[blue]Exporting memories for category: {category}[/blue]")
            file_path = exporter.export_category(category)
            console.print(f"[green]Exported to: {file_path}[/green]")
        else:
            console.print(f"[blue]Exporting memories (grouped by {group_by})...[/blue]")
            files = exporter.export()

            if not files:
                console.print("[yellow]No memories found to export[/yellow]")
                return

            for file_path in files:
                console.print(f"[green]Exported to: {file_path}[/green]")

    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise SystemExit(1) from None
    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error exporting memories: {e}[/red]")
        raise SystemExit(1) from None


@cli.command("prune-stale")
@click.option(
    "--max-age-days",
    default=90,
    show_default=True,
    help="Delete memories older than N days",
)
@click.option("--source-tool", "-s", help="Only prune this source")
@click.option(
    "--dry-run/--no-dry-run",
    is_flag=True,
    default=True,
    show_default=True,
    help="Preview without deleting",
)
@click.option(
    "--execute",
    is_flag=True,
    help="Actually delete stale memories (overrides --dry-run)",
)
def prune_stale(
    max_age_days: int,
    source_tool: str | None,
    dry_run: bool,
    execute: bool,
) -> None:
    """Delete memories older than a given age."""
    try:
        config = load_config()
        storage = Storage(config=config)

        if execute:
            dry_run = False

        mode = "[yellow]DRY RUN[/yellow]" if dry_run else "[red]LIVE[/red]"
        console.print(f"[blue]Pruning memories older than {max_age_days} days ({mode})...[/blue]")

        counts = storage.count_stale_memories(max_age_days, source_tool)
        if not counts:
            console.print("[green]No stale memories found.[/green]")
            return

        total = sum(r["count"] for r in counts)
        table = Table(title="Stale Memories", show_header=True, header_style="bold magenta")
        table.add_column("Source", style="cyan")
        table.add_column("Category", style="green")
        table.add_column("Count", style="yellow")
        for r in counts:
            table.add_row(r["source_tool"], r["category"], str(r["count"]))
        table.add_row("[bold]Total[/bold]", "", f"[bold]{total}[/bold]")
        console.print(table)

        if dry_run:
            console.print(
                "\n[yellow]This was a dry run. Use --execute to actually delete.[/yellow]"
            )
        else:
            deleted = storage.delete_stale_memories(max_age_days, source_tool)
            console.print(f"\n[green]Deleted {deleted} stale memories.[/green]")

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except Exception as e:
        console.print(f"[red]Error pruning: {e}[/red]")
        raise SystemExit(1) from None


@cli.command()
@click.argument("memory_id", type=int)
def promote(memory_id: int) -> None:
    """Promote a memory to the 'promoted' category."""
    try:
        config = load_config()
        storage = Storage(config=config)

        memory = storage.get_memory_by_id(memory_id)
        if not memory:
            console.print(f"[red]Memory #{memory_id} not found.[/red]")
            raise SystemExit(1)

        old_category = memory["category"]
        if old_category == "promoted":
            console.print(f"[yellow]Memory #{memory_id} is already promoted.[/yellow]")
            return

        storage.update_memory_category(memory_id, "promoted")
        console.print(f"[green]Promoted memory #{memory_id}:[/green]")
        console.print(f"  [dim]{old_category}[/dim] -> [bold green]promoted[/bold green]")
        console.print(f"  {memory['content']}")

    except FileNotFoundError:
        console.print("[red]Config not found. Run 'oghma init' first.[/red]")
        raise SystemExit(1) from None
    except SystemExit:
        raise
    except Exception as e:
        console.print(f"[red]Error promoting memory: {e}[/red]")
        raise SystemExit(1) from None


def main() -> None:
    cli()
