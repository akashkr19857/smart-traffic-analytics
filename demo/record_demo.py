"""Record the demo video of the real running app (start run_demo.bat and let the cameras run ~1 minute first).

    python demo/record_demo.py        → demo/demo.mp4, demo/demo.gif, demo/screenshots/*.png
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from recorder import record  # noqa: E402

WEB = "http://localhost:3400"
SHOTS = HERE / "screenshots"


async def shot(page, name: str) -> None:
    await page.screenshot(path=str(SHOTS / f"{name}.png"))


async def steps(page, caption) -> None:
    SHOTS.mkdir(exist_ok=True)
    await page.goto(WEB)
    await page.wait_for_selector("[data-testid=main-stream]")
    await caption("Live 4K highway: YOLO11 detection + ByteTrack tracking + speed, on a laptop GPU", 6000)
    await caption("Each vehicle: class, track ID, speed in km/h; vehicles counted at the yellow line", 5000)
    await shot(page, "01_live_highway")
    await page.click("[data-testid=camera-highway-incidents]")
    await caption("SIMULATED test feed: time-reversed footage → every vehicle is a wrong-way driver", 7000)
    await shot(page, "02_wrong_way")
    await caption("Red zones = hard shoulder (no parking); a car pasted there triggers illegal parking", 6000)
    await page.click("[data-testid=camera-expressway-city]")
    await caption("City expressway: counting and classifying dense traffic", 5000)
    await shot(page, "03_city")

    await page.goto(f"{WEB}/events")
    await page.wait_for_selector("[data-testid=event-list]")
    await caption("Violations with evidence snapshots: wrong-way, illegal parking, speeding", 4000)
    rows = page.locator("[data-testid=event-row]")
    for i in range(min(await rows.count(), 3)):
        await rows.nth(i).click()
        await page.wait_for_timeout(1700)
    await shot(page, "04_violations")

    await page.goto(f"{WEB}/anpr")
    await page.wait_for_selector("[data-testid=plate-reads]")
    await caption("ANPR: plate detector + OCR, voted across frames of the same tracked vehicle", 5000)
    await shot(page, "05_anpr")
    await page.fill("[data-testid=plate-search]", "412")
    await page.click("[data-testid=plate-search-go]")
    await page.wait_for_timeout(1500)
    await caption("Search any (partial) plate across all cameras", 3500)
    await page.fill("[data-testid=watch-plate]", "NZ412GC")
    await page.click("[data-testid=watch-add]")
    await caption("Watchlist: a listed plate raises a critical alert on every camera", 3500)

    await page.goto(f"{WEB}/analytics")
    await page.wait_for_selector("[data-testid=analytics-kpis]")
    await page.wait_for_timeout(2500)
    await caption("Traffic analytics: volumes by class, speed distribution, passages", 4500)
    await shot(page, "06_analytics")
    await page.goto(f"{WEB}/benchmarks")
    await page.wait_for_selector("[data-testid=bench-metric]")
    await caption("Every number is measured: detector mAP, OCR ablation, voting gain, FPS", 5500)
    await shot(page, "07_benchmarks")
    await caption("")


if __name__ == "__main__":
    record(steps, out_dir=HERE, title="AI Smart Traffic Analytics", subtitle="ANPR · YOLO11 · ByteTrack · speed · wrong-way · illegal parking")
