"""LangChain tools used by the LangGraph inference workflow."""

from __future__ import annotations

from typing import Any, Optional

import httpx
from langchain.tools import tool
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class LiveNvdToolInput(BaseModel):
    """Structured input for the live NVD fallback tool.

    Input:
        Carries the rewritten user query and an optional NVD page-size cap.
    Output:
        Validates tool arguments before the external API call is made.
    Security context:
        Keeps the NVD lookup surface bounded to a keyword search and a small
        result window so inference remains read-only and predictable.
    """

    query: str = Field(
        default="",
        description="Keyword query used against the live NVD CVE API.",
    )
    results_per_page: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of live NVD CVE records to fetch.",
    )


@tool(args_schema=LiveNvdToolInput)
async def fetch_live_nvd_tool(query: str = "", results_per_page: int = 5) -> list[Document]:
    """Fetch live NVD CVE records and normalize them as documents.

    Input:
        A keyword query and a bounded result limit.
    Output:
        Returns LangChain `Document` objects derived from NVD CVE items.
    Security context:
        Uses only read-only NVD API access and preserves CVE identifiers in
        metadata so fallback evidence remains attributable.
    """

    params: dict[str, Any] = {"resultsPerPage": results_per_page}
    if query:
        params["keywordSearch"] = query

    async with httpx.AsyncClient(
        base_url="https://services.nvd.nist.gov",
        timeout=30.0,
    ) as client:
        response = await client.get("/rest/json/cves/2.0", params=params)
        response.raise_for_status()
        payload = response.json()

    return [
        _nvd_item_to_document(item)
        for item in payload.get("vulnerabilities", [])
    ]


def _nvd_item_to_document(item: dict[str, Any]) -> Document:
    """Convert an NVD API vulnerability item into a LangChain document.

    Input:
        A single vulnerability object from the NVD `cves/2.0` API response.
    Output:
        Returns a LangChain `Document` for fallback evidence generation.
    Security context:
        Preserves source identifiers and CVSS data so fallback answers remain
        attributable to authoritative NVD records.
    """

    cve = item.get("cve", {})
    cve_id = str(cve.get("id", "unknown-cve"))

    description = ""
    for entry in cve.get("descriptions", []):
        if entry.get("lang") == "en":
            description = str(entry.get("value", ""))
            break

    cvss_score: Optional[float] = None
    metrics = cve.get("metrics", {})
    for key in ("cvssMetricV31", "cvssMetricV30", "cvssMetricV2"):
        metric_list = metrics.get(key, [])
        if metric_list:
            base_score = metric_list[0].get("cvssData", {}).get("baseScore")
            if base_score is not None:
                cvss_score = float(base_score)
                break

    return Document(
        page_content=description,
        metadata={
            "source": "nvd_live",
            "cve_id": cve_id,
            "cvss": cvss_score,
        },
    )
