"""
Convierte docs/INFORME.md a un HTML profesional y autocontenido (docs/INFORME.html)
listo para ver o exportar a PDF (Ctrl+P / Imprimir en PDF).
Incrusta estilos CSS formales, KaTeX para fórmulas matemáticas e imágenes en Base64.
"""

import base64
from pathlib import Path
import re
import markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
MD_FILE = DOCS_DIR / "INFORME.md"
HTML_FILE = DOCS_DIR / "INFORME.html"


def embed_images_as_base64(html_content: str, base_dir: Path) -> str:
    """Reemplaza rutas relativas de imágenes con Data URIs en base64."""
    img_pattern = re.compile(r'<img [^>]*src="([^"]+)"[^>]*>')

    def replacer(match):
        full_tag = match.group(0)
        src = match.group(1)
        if src.startswith("data:") or src.startswith("http"):
            return full_tag
        img_path = (base_dir / src).resolve()
        if not img_path.exists():
            return full_tag
        mime = "image/png" if img_path.suffix.lower() == ".png" else "image/jpeg"
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        data_uri = f"data:{mime};base64,{encoded}"
        return full_tag.replace(src, data_uri)

    return img_pattern.sub(replacer, html_content)


def main():
    if not MD_FILE.exists():
        print(f"Error: No se encontró {MD_FILE}")
        return

    with open(MD_FILE, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Pre-procesamiento de bloques KaTeX para evitar que markdown rompa las fórmulas
    # Convierte $$ ... $$ y $ ... $
    # KaTeX en el navegador procesará automáticamente con renderMathInElement
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc"]
    )

    # Reemplazar imágenes por base64
    html_body = embed_images_as_base64(html_body, DOCS_DIR)

    full_html = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Informe Final - Proyecto 3 CUDA (CC3069)</title>
    <!-- KaTeX CDN -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
    <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
        onload="renderMathInElement(document.body, {{
            delimiters: [
                {{left: '$$', right: '$$', display: true}},
                {{left: '$', right: '$', display: false}}
            ]
        }});"></script>
    <style>
        @page {{
            size: letter;
            margin: 2.2cm 2cm;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.6;
            color: #24292f;
            background-color: #ffffff;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px 30px;
        }}
        h1, h2, h3, h4 {{
            color: #0969da;
            font-weight: 600;
            margin-top: 1.4em;
            margin-bottom: 0.6em;
            page-break-after: avoid;
        }}
        h1 {{
            font-size: 20pt;
            color: #1f2328;
            border-bottom: 2px solid #d0d7de;
            padding-bottom: 8px;
            margin-top: 0.5em;
        }}
        h2 {{
            font-size: 15pt;
            border-bottom: 1px solid #d0d7de;
            padding-bottom: 5px;
            color: #1a7f37;
        }}
        h3 {{
            font-size: 12.5pt;
            color: #0969da;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        th, td {{
            border: 1px solid #d0d7de;
            padding: 8px 12px;
            text-align: left;
        }}
        th {{
            background-color: #f6f8fa;
            font-weight: 600;
            color: #1f2328;
        }}
        tr:nth-child(even) {{
            background-color: #fbfcfd;
        }}
        blockquote {{
            margin: 16px 0;
            padding: 10px 16px;
            color: #57606a;
            background-color: #f6f8fa;
            border-left: 4px solid #0969da;
            page-break-inside: avoid;
        }}
        pre, code {{
            font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
            font-size: 9pt;
        }}
        pre {{
            background-color: #f6f8fa;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #d0d7de;
            overflow-x: auto;
            page-break-inside: avoid;
        }}
        p > code {{
            background-color: rgba(175, 184, 193, 0.2);
            padding: 0.2em 0.4em;
            border-radius: 4px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 18px auto;
            border-radius: 6px;
            border: 1px solid #d0d7de;
            page-break-inside: avoid;
        }}
        hr {{
            border: 0;
            height: 1px;
            background-color: #d0d7de;
            margin: 24px 0;
        }}
        @media print {{
            body {{
                max-width: 100%;
                padding: 0;
            }}
            .no-print {{
                display: none;
            }}
        }}
        .print-banner {{
            background: #e7f5ff;
            border: 1px solid #a5d8ff;
            border-radius: 6px;
            padding: 10px 16px;
            margin-bottom: 24px;
            font-size: 10pt;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .print-btn {{
            background: #0969da;
            color: white;
            border: none;
            padding: 6px 14px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: 600;
        }}
        .print-btn:hover {{
            background: #0856b3;
        }}
    </style>
</head>
<body>
    <div class="print-banner no-print">
        <span>📄 <strong>Versión formateada del Informe Final</strong> (Listo para ver o exportar a PDF).</span>
        <button class="print-btn" onclick="window.print()">Imprimir / Guardar en PDF</button>
    </div>
    {html_body}
</body>
</html>
"""

    with open(HTML_FILE, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Informe renderizado exitosamente en: {HTML_FILE}")


if __name__ == "__main__":
    main()
