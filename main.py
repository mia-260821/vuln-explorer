"""CLI entrypoint that invokes the LangGraph workflow."""

from __future__ import annotations

import argparse
import asyncio
import json

from agents.usecase import run_agent_graph
from config import AppConfig
from ingestion.usecase import build_nvd_harvester


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the LangGraph workflow.

    Input:
        Reads arguments from the current CLI invocation.
    Output:
        Returns a namespace describing the requested workflow mode and filters.
    Security context:
        Keeps workflow mode and query scope explicit so the graph executes only
        the intended ingestion or analysis path.
    """

    parser = argparse.ArgumentParser(description="vulne-explorer CLI")
    parser.add_argument("--ingest", action="store_true", help="Run standalone ingestion into Qdrant")
    parser.add_argument("--question", type=str, default=None, help="Search question for vulnerability analysis")
    parser.add_argument("--os", dest="operating_system", type=str, default=None, help="Operating system filter")
    parser.add_argument(
        "--stack",
        nargs="*",
        default=[],
        help="Optional software stack filters used during retrieval",
    )
    return parser.parse_args()


async def run_ingestion(config: AppConfig) -> None:
    """Run the standalone ingestion flow for NVD documents.

    Input:
        Application configuration with NVD, embeddings, and Qdrant settings.
    Output:
        Fetches NVD documents and incrementally upserts them into Qdrant.
    Security context:
        Keeps ingestion as a standalone write-time operation separate from the
        read-only LangGraph inference workflow.
    """

    harvester = await build_nvd_harvester(config)
    documents = await harvester.fetch_documents()

    batch_size = 10
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        await harvester.sync_documents(documents=batch)
    return documents


async def run() -> int:
    """Invoke the compiled LangGraph workflow from the CLI.

    Input:
        Uses parsed CLI arguments and application configuration.
    Output:
        Returns a process exit code and prints the result as JSON.
    Security context:
        Keeps ingestion outside LangGraph and constrains LangGraph execution to
        the read-only inference path.
    """

    args = parse_args()
    config = AppConfig()
    config.ensure_directories()

    if args.ingest:
        documents = await run_ingestion(config)
        print(json.dumps({"ingested_chunks": len(documents)}, indent=2))
        return 0

    if args.question:
        result = await run_agent_graph(
            state={
                "query": args.question,
                "operating_system": args.operating_system,
                "software_stack": args.stack,
            },
            config=config,
        )
        print(json.dumps({"answer": result["final_answer"]}, indent=2))
        return 0

    print("No action specified. Use --ingest or --question.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
