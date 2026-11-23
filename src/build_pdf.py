"""Build COMPTE_RENDU.pdf from COMPTE_RENDU.md (Markdown -> HTML -> PDF with headless Edge/Chrome)."""
import re
import shutil
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
md = (ROOT / "COMPTE_RENDU.md").read_text(encoding="utf-8")
md = re.sub(r"\$\sqrt\{d_k\}\$", "√d<sub>k</sub>", md)
md = re.sub(r"\$([^$]+)\$", r"\1", md)
body = markdown.markdown(md, extensions=["tables"])
html = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8"><title>Compte-rendu TP4</title>
<style>
@page {{ size: A4; margin: 16mm 15mm; }}
body {{ font-family: 'Segoe UI', Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #1a1a1a; }}
h1 {{ font-size: 17pt; border-bottom: 2px solid #1f4e79; padding-bottom: 4px; color: #1f4e79; }}
h2 {{ font-size: 13pt; color: #1f4e79; margin-top: 18px; border-bottom: 1px solid #ccd; padding-bottom: 2px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 9.5pt; }}
th, td {{ border: 1px solid #b8c2cc; padding: 4px 7px; text-align: left; }}
th {{ background: #e8eef5; }}
code {{ background: #f2f4f7; padding: 0 3px; border-radius: 3px; font-size: 9pt; }}
p, li {{ text-align: justify; }}
</style></head><body>{body}</body></html>"""
out_html = ROOT / "results" / "compte_rendu.html"
out_html.write_text(html, encoding="utf-8")
browser = next(p for p in [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                           r"C:\Program Files\Google\Chrome\Application\chrome.exe", shutil.which("chromium")] if p and Path(p).exists())
pdf = ROOT / "COMPTE_RENDU.pdf"
subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={pdf}", out_html.as_uri()], check=True, timeout=120)
print("PDF:", pdf, pdf.stat().st_size, "bytes")
