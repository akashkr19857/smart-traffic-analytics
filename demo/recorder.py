"""Record a real demo video of the running dashboard: Playwright records the browser, ffmpeg converts it.

Used by demo/record_demo.py. Needs `pip install playwright && python -m playwright install chromium` and ffmpeg
(on PATH, or set the FFMPEG environment variable to its full path).

Output: demo/demo.mp4 (1280x720, H.264) and demo/demo.gif (a short loop for the README).
"""
from __future__ import annotations

import asyncio
import glob
import os
import shutil
import subprocess
from pathlib import Path
from typing import Awaitable, Callable

CAPTION_JS = """
(text) => {
  let el = document.getElementById('__demo_caption');
  if (!el) {
    el = document.createElement('div');
    el.id = '__demo_caption';
    Object.assign(el.style, {position:'fixed', left:'50%', bottom:'28px', transform:'translateX(-50%)',
      background:'rgba(10,14,20,.86)', color:'#fff', font:'600 20px system-ui, sans-serif',
      padding:'12px 22px', borderRadius:'10px', zIndex:2147483647, maxWidth:'80%', textAlign:'center',
      boxShadow:'0 6px 24px rgba(0,0,0,.35)', transition:'opacity .3s', pointerEvents:'none'});
    document.body.appendChild(el);
  }
  el.textContent = text; el.style.opacity = text ? '1' : '0';
}
"""

TITLE_HTML = """<html><body style="margin:0;height:100vh;display:grid;place-items:center;background:#0b1220;
color:#fff;font-family:system-ui,sans-serif"><div style="text-align:center"><div style="font-size:44px;font-weight:800">{title}</div>
<div style="margin-top:14px;font-size:22px;color:#9fb3c8">{subtitle}</div>
<div style="margin-top:28px;font-size:18px;color:#5eead4">Built by Akash Kumar</div></div></body></html>"""


def ffmpeg_path() -> str:
    found = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not found:  # winget install location on Windows
        hits = glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*FFmpeg*\*\bin\ffmpeg.exe"))
        found = hits[0] if hits else None
    if not found:
        raise FileNotFoundError("ffmpeg not found: install it or set FFMPEG=<path to ffmpeg>")
    return found


async def _run(steps, out_dir: Path, title: str, subtitle: str, width: int, height: int) -> Path:  # noqa: ANN001
    from playwright.async_api import async_playwright

    raw_dir = out_dir / "_raw"
    shutil.rmtree(raw_dir, ignore_errors=True)
    raw_dir.mkdir(parents=True)
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        ctx = await browser.new_context(viewport={"width": width, "height": height}, record_video_dir=str(raw_dir),
                                        record_video_size={"width": width, "height": height})
        page = await ctx.new_page()
        await page.set_content(TITLE_HTML.format(title=title, subtitle=subtitle))
        await page.wait_for_timeout(3500)

        async def caption(text: str, hold_ms: int = 0) -> None:
            await page.evaluate(CAPTION_JS, text)
            if hold_ms:
                await page.wait_for_timeout(hold_ms)

        await steps(page, caption)
        await page.set_content(TITLE_HTML.format(title=title, subtitle="Code on GitHub: github.com/akashkr19857"))
        await page.wait_for_timeout(3000)
        await ctx.close()
        await browser.close()
    webms = sorted(raw_dir.glob("*.webm"))
    if not webms:
        raise RuntimeError("Playwright produced no video")
    return webms[-1]


def record(steps: Callable[..., Awaitable[None]], out_dir: Path, title: str, subtitle: str = "",
           width: int = 1280, height: int = 720) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    webm = asyncio.run(_run(steps, out_dir, title, subtitle, width, height))
    ff = ffmpeg_path()
    mp4, gif = out_dir / "demo.mp4", out_dir / "demo.gif"
    subprocess.run([ff, "-y", "-i", str(webm), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", "-movflags", "+faststart",
                    str(mp4)], check=True, capture_output=True)
    subprocess.run([ff, "-y", "-ss", "3.5", "-t", "12", "-i", str(mp4), "-vf",
                    "fps=8,scale=880:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse", str(gif)],
                   check=True, capture_output=True)
    shutil.rmtree(webm.parent, ignore_errors=True)
    print(f"demo video: {mp4}\ndemo gif:   {gif}")
    return mp4
