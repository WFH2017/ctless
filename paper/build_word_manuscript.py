"""Build synchronized professional DOCX and XeLaTeX drafts from main_manuscript.md.

Features:
- Academic Three-Line Tables (top rule 1.5pt, header rule 0.75pt, bottom rule 1.5pt)
- Table properties: tblHeader on row 0, cantSplit on all rows
- Figure & Caption grouping with keep_with_next
- Inline typography: bold, italic, code, math symbols
- Professional running headers and page numbering footers
- Synchronized XeLaTeX generation
"""
from pathlib import Path
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

ROOT = Path(__file__).resolve().parent
OUTPUT_DOCX = ROOT / 'CTless_SPECT_Universal_Reconstruction_manuscript.docx'
OUTPUT_TEX = ROOT / 'main_manuscript.tex'

def clean_math_text(math_text: str) -> str:
    """Format LaTeX math string into clean unicode representation for Word."""
    m = math_text.strip('$')
    replacements = [
        (r'\mu', 'μ'), (r'\times', '×'), (r'\pm', '±'),
        (r'\le', '≤'), (r'\ge', '≥'), (r'\hat', ''),
        (r'\text', ''), (r'\mathbb', ''), (r'\mathcal', ''),
        (r'\Delta', 'Δ'), (r'\lambda', 'λ'), (r'\tau', 'τ'),
        (r'\delta', 'δ'), (r'\odot', '⊙'), (r'\Omega', 'Ω'),
        (r'\in', '∈'), (r'\mathbb{R}', 'ℝ'), (r'\approx', '≈'),
        (r'\sim', '~'), (r'\infty', '∞'),
        ('{', ''), ('}', ''), ('\\', '')
    ]
    for old, new in replacements:
        m = m.replace(old, new)
    return m.strip()

def add_formatted_runs(paragraph, text: str, base_font='Times New Roman', base_size=Pt(11)):
    """Parse inline markdown (bold, italic, code, math) into distinct runs."""
    pattern = re.compile(r'(\*\*.*?\*\*|\*.*?\*|\`.*?\`|\$\$.*?\$\$|\$.*?\$|\[To be completed.*?\]|\[.*?to be finalized.*?\])')
    tokens = pattern.split(text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = paragraph.add_run(token[2:-2])
            run.bold = True
            run.font.name = base_font
            run.font.size = base_size
        elif token.startswith('*') and token.endswith('*') and not token.startswith('**'):
            run = paragraph.add_run(token[1:-1])
            run.italic = True
            run.font.name = base_font
            run.font.size = base_size
        elif token.startswith('`') and token.endswith('`'):
            run = paragraph.add_run(token[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(160, 40, 40)
        elif (token.startswith('$$') and token.endswith('$$')) or (token.startswith('$') and token.endswith('$')):
            math_str = clean_math_text(token)
            run = paragraph.add_run(math_str)
            run.font.name = 'Cambria Math'
            run.font.size = Pt(10.5)
            run.italic = True
        elif 'to be' in token.lower():
            run = paragraph.add_run(token)
            run.font.name = base_font
            run.font.size = base_size
            run.font.italic = True
            run.font.color.rgb = RGBColor(110, 110, 110)
        else:
            run = paragraph.add_run(token)
            run.font.name = base_font
            run.font.size = base_size

def style_three_line_table(table):
    """Apply international academic three-line table format."""
    # Top and bottom 1.5pt borders, no vertical borders, no inside horizontal borders
    tblBorders = parse_xml(r'''
        <w:tblBorders %s>
            <w:top w:val="single" w:sz="12" w:space="0" w:color="1A2B4C"/>
            <w:left w:val="none"/>
            <w:bottom w:val="single" w:sz="12" w:space="0" w:color="1A2B4C"/>
            <w:right w:val="none"/>
            <w:insideH w:val="none"/>
            <w:insideV w:val="none"/>
        </w:tblBorders>
    ''' % nsdecls('w'))
    table._tbl.tblPr.append(tblBorders)
    table.alignment = docx.enum.table.WD_TABLE_ALIGNMENT.CENTER

    # Style header row
    hdr_row = table.rows[0]
    trPr0 = hdr_row._tr.get_or_add_trPr()
    trPr0.append(parse_xml(r'<w:tblHeader %s/>' % nsdecls('w')))
    trPr0.append(parse_xml(r'<w:cantSplit %s/>' % nsdecls('w')))

    for cell in hdr_row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        # Header bottom line (0.75 pt)
        tcPr.append(parse_xml(r'''
            <w:tcBorders %s>
                <w:bottom w:val="single" w:sz="6" w:space="0" w:color="1A2B4C"/>
            </w:tcBorders>
        ''' % nsdecls('w')))
        tcPr.append(parse_xml(r'<w:shd %s w:fill="F1F4F8"/>' % nsdecls('w')))
        # Set cell vertical alignment to center
        tcPr.append(parse_xml(r'<w:vAlign %s w:val="center"/>' % nsdecls('w')))

    # Style all data rows
    for r_idx, row in enumerate(table.rows[1:], start=1):
        trPr = row._tr.get_or_add_trPr()
        trPr.append(parse_xml(r'<w:cantSplit %s/>' % nsdecls('w')))
        # subtle zebra shading for readability on large tables
        if r_idx % 2 == 1:
            for cell in row.cells:
                tcPr = cell._tc.get_or_add_tcPr()
                tcPr.append(parse_xml(r'<w:shd %s w:fill="FAFBFC"/>' % nsdecls('w')))

def plain(s):
    return re.sub(r'\*\*(.*?)\*\*', r'\1', s).replace('`', '').replace('$', '')

def tex(s):
    replacements = {
        '\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
        '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        'μ': r'$\mu$', '±': r'$\pm$', '×': r'$\times$', '≤': r'$\le$', '≥': r'$\ge$'
    }
    res = plain(s)
    for c, r in replacements.items():
        res = res.replace(c, r)
    return res

def build():
    lines = (ROOT / 'main_manuscript.md').read_text(encoding='utf-8').splitlines()
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(0.85)
    section.bottom_margin = Inches(0.85)
    section.left_margin = Inches(0.85)
    section.right_margin = Inches(0.85)

    # Professional Running Header
    header = section.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hrun = hp.add_run('Tissue-Balanced Attenuation Map Estimation from Dual-Window SPECT')
    hrun.font.name = 'Times New Roman'
    hrun.font.size = Pt(8.5)
    hrun.font.italic = True
    hrun.font.color.rgb = RGBColor(120, 120, 120)

    # Footer with Page Number
    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    frun = fp.add_run('Page ')
    frun.font.name = 'Times New Roman'
    frun.font.size = Pt(9)
    frun.font.color.rgb = RGBColor(120, 120, 120)
    fp._p.append(parse_xml(r'<w:fldSimple %s w:instr="PAGE"/>' % nsdecls('w')))

    # Global Style Setup
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(11)
    normal_style.paragraph_format.space_after = Pt(4)
    normal_style.paragraph_format.line_spacing = 1.15

    latex = [
        r'\documentclass[11pt]{article}',
        r'\usepackage[margin=0.8in]{geometry}',
        r'\usepackage{fontspec}',
        r'\setmainfont{DejaVu Serif}',
        r'\usepackage{graphicx,longtable,array,hyperref,booktabs,amsmath}',
        r'\hypersetup{hidelinks}',
        r'\begin{document}'
    ]

    i = 0
    in_references = False

    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue

        # Markdown Table Parsing
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].startswith('|'):
                cells = [c.strip() for c in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    rows.append(cells)
                i += 1

            if not rows:
                continue

            num_cols = len(rows[0])
            table = doc.add_table(rows=len(rows), cols=num_cols)
            style_three_line_table(table)

            for r_idx, row in enumerate(rows):
                for c_idx, cell_text in enumerate(row):
                    cell = table.cell(r_idx, c_idx)
                    cell.text = ""  # clear default text
                    p = cell.paragraphs[0]
                    p.paragraph_format.space_before = Pt(3)
                    p.paragraph_format.space_after = Pt(3)
                    p.paragraph_format.line_spacing = 1.05

                    if c_idx == 0:
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    else:
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    is_header = (r_idx == 0)
                    font_size = Pt(9) if is_header else Pt(8.5)
                    add_formatted_runs(p, cell_text, base_size=font_size)
                    if is_header:
                        for run in p.runs:
                            run.bold = True
                            run.font.color.rgb = RGBColor(20, 35, 60)

            # Add spacing after table
            sp_p = doc.add_paragraph()
            sp_p.paragraph_format.space_before = Pt(2)
            sp_p.paragraph_format.space_after = Pt(6)

            # LaTeX Table Generation
            spec = 'p{0.32\\linewidth}' + 'p{0.14\\linewidth}' * (num_cols - 1)
            latex.append(r'\begingroup\small\setlength{\tabcolsep}{3pt}\begin{longtable}{' + spec + '}')
            latex.append(r'\toprule')
            for r_idx, row in enumerate(rows):
                latex.append(' & '.join(tex(c) for c in row) + r' \\')
                if r_idx == 0:
                    latex.append(r'\midrule')
            latex.append(r'\bottomrule')
            latex.append(r'\end{longtable}\endgroup')
            continue

        # Figure Images
        match = re.match(r'!\[(.*?)\]\((.*?)\)', line)
        if match:
            path = ROOT / match[2]
            assert path.exists(), f"Image path does not exist: {path}"
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(path), width=Inches(6.4))
            latex.append(r'\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{' + match[2] + r'}\end{figure}')
            i += 1
            continue

        # Figure / Table Captions
        if line.startswith('**Figure ') or line.startswith('**Table ') or line.startswith('**Supplementary Figure '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(10)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.left_indent = Inches(0.15)
            p.paragraph_format.right_indent = Inches(0.15)
            add_formatted_runs(p, line, base_size=Pt(9.5))
            latex.append(r'\begin{quote}\small\textbf{' + tex(line) + r'}\end{quote}')
            i += 1
            continue

        # Document Title (# Title)
        if line.startswith('# '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(10)
            run = p.add_run(plain(line[2:]))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(18)
            run.bold = True
            run.font.color.rgb = RGBColor(15, 30, 60)
            latex.append(r'\title{\textbf{' + tex(line[2:]) + r'}}\author{Fenghua Weng et al.}\date{}\maketitle')
            i += 1
            continue

        # Section Heading 1 (## Heading)
        if line.startswith('## '):
            heading_text = line[3:]
            if heading_text.strip().lower() == 'references':
                in_references = True
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(5)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(plain(heading_text))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(13)
            run.bold = True
            run.font.color.rgb = RGBColor(20, 45, 80)
            latex.append(r'\section*{' + tex(heading_text) + '}')
            i += 1
            continue

        # Subsection Heading 2 (### Heading)
        if line.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(11)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(plain(line[4:]))
            run.font.name = 'Times New Roman'
            run.font.size = Pt(11.5)
            run.bold = True
            run.font.color.rgb = RGBColor(30, 60, 100)
            latex.append(r'\subsection*{' + tex(line[4:]) + '}')
            i += 1
            continue

        # Math Display Blocks ($$ ... $$)
        if line.startswith('$$') and line.endswith('$$'):
            math_content = clean_math_text(line)
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(math_content)
            run.font.name = 'Cambria Math'
            run.font.size = Pt(11)
            run.italic = True
            latex.append(r'\begin{equation*}' + line.strip('$') + r'\end{equation*}')
            i += 1
            continue

        # Indented Math / Code
        if line.startswith('    '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(plain(line.strip()))
            run.font.name = 'Cambria Math'
            run.font.size = Pt(10)
            run.italic = True
            latex.append(r'\begin{quote}\small\texttt{' + tex(line.strip()) + r'}\end{quote}')
            i += 1
            continue

        # Lists (- bullet or 1. item)
        if line.startswith('- ') or re.match(r'^\d+\.\s', line):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.25)
            p.paragraph_format.space_before = Pt(1)
            p.paragraph_format.space_after = Pt(3)
            p.paragraph_format.line_spacing = 1.15
            add_formatted_runs(p, line, base_size=Pt(10.5))
            latex.append(r'\begin{itemize}\item ' + tex(line[2:]) + r'\end{itemize}')
            i += 1
            continue

        # Standard Body Paragraphs
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(5)
        p.paragraph_format.line_spacing = 1.15
        
        # In references section, use smaller font and hanging indent
        if in_references and re.match(r'^\d+\.\s', line):
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.first_line_indent = Inches(-0.3)
            p.paragraph_format.space_after = Pt(3)
            add_formatted_runs(p, line, base_size=Pt(9.5))
        else:
            add_formatted_runs(p, line, base_size=Pt(11))
            
        latex.append(tex(line) + r'\par' + '\n')
        i += 1

    latex.append(r'\end{document}')
    doc.save(OUTPUT_DOCX)
    OUTPUT_TEX.write_text('\n'.join(latex) + '\n', encoding='utf-8')
    print(f'Successfully compiled {OUTPUT_DOCX.name} and {OUTPUT_TEX.name} with publication-quality formatting.')

if __name__ == '__main__':
    build()
