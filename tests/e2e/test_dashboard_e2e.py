"""End-to-end test against the real running app (API :8400 with GPU pipelines + dashboard :3400).

Start run_demo.bat and let the cameras run ~1 minute, then:  python -m pytest tests/e2e -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
import urllib.request

import pytest

WEB = os.environ.get("TRAFFIC_WEB", "http://localhost:3400")
API = os.environ.get("TRAFFIC_API", "http://localhost:8400")


def _get(url: str):  # noqa: ANN202
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return json.loads(r.read()) if "api" in url else r.status
    except Exception:
        return None


pytestmark = pytest.mark.skipif(not (_get(WEB) and _get(f"{API}/api/health")), reason="app is not running on :3400/:8400")


def test_live_pipeline_produces_data():
    cams = _get(f"{API}/api/cameras")
    assert all(c["stats"]["status"] == "running" for c in cams)
    assert sum(c["stats"].get("passed", 0) for c in cams) > 0
    assert any(e["type"] == "wrong_way" for e in _get(f"{API}/api/events?type=wrong_way&camera_id=highway-incidents"))
    with urllib.request.urlopen(f"{API}/api/cameras/highway-d1/snapshot.jpg", timeout=5) as r:
        assert r.headers["content-type"] == "image/jpeg" and len(r.read()) > 10_000


def test_dashboard_pages():
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as p:
        b = p.chromium.launch()
        page = b.new_page(viewport={"width": 1440, "height": 900})
        page.goto(WEB)
        expect(page.get_by_test_id("main-stream")).to_be_visible()
        expect(page.get_by_test_id("live-stats")).to_contain_text("Vehicles counted")
        page.goto(f"{WEB}/events")
        page.get_by_test_id("event-row").first.click()
        expect(page.get_by_test_id("evidence")).to_be_visible()
        page.goto(f"{WEB}/anpr")
        page.get_by_test_id("watch-plate").fill("E2E123")
        page.get_by_test_id("watch-add").click()
        expect(page.get_by_test_id("watchlist")).to_contain_text("E2E123")
        page.goto(f"{WEB}/benchmarks")
        expect(page.get_by_test_id("bench-metric").first).to_be_visible()
        b.close()
    urllib.request.urlopen(urllib.request.Request(f"{API}/api/watchlist/E2E123", method="DELETE"), timeout=5)
