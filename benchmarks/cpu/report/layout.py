"""Native Word styles, tables, equations and links for the experiment report."""

from html import escape
from os import getenv
from pathlib import Path
import re

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.shared import Inches, Pt, RGBColor
from lxml import etree


SUBSCRIPT = dict(zip('₀₁₂₃₄₅₆₇₈₉₊₋ₐₑᵢⱼₖₗₘₙₒₚᵣₛₜ', '0123456789+−aeijklmnoprst'))
SUPERSCRIPT = dict(zip('⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻', '0123456789+−'))
CHEM_TOKENS = re.compile('([' + ''.join(SUBSCRIPT) + ']+|[' + ''.join(SUPERSCRIPT) + ']+)')
HEADER_FILL = '25445B'
CAPTION_COLOR = '303E49'
MML_NS = 'http://www.w3.org/1998/Math/MathML'


def font(style, size, *, east='Songti SC', latin='Arial', bold=False, color='000000'):
    style.font.name = latin
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor.from_string(color)
    rfonts = style.element.get_or_add_rPr().rFonts
    for key, val in (('ascii', latin), ('hAnsi', latin), ('eastAsia', east), ('cs', latin)):
        rfonts.set(qn(f'w:{key}'), val)
    for key in ('asciiTheme', 'hAnsiTheme', 'eastAsiaTheme', 'cstheme'):
        rfonts.attrib.pop(qn(f'w:{key}'), None)


def text(paragraph, value, bold=False):
    for part in CHEM_TOKENS.split(str(value)):
        if part:
            if part[0] in SUBSCRIPT:
                run = paragraph.add_run(''.join(SUBSCRIPT[c] for c in part))
                run.font.subscript = True
            elif part[0] in SUPERSCRIPT:
                run = paragraph.add_run(''.join(SUPERSCRIPT[c] for c in part))
                run.font.superscript = True
            else:
                run = paragraph.add_run(part)
            if bold:
                run.bold = True


def field(paragraph, instruction):
    node = OxmlElement('w:fldSimple')
    node.set(qn('w:instr'), instruction)
    run = OxmlElement('w:r')
    val = OxmlElement('w:t')
    val.text = '1'
    run.append(val)
    node.append(run)
    paragraph._p.append(node)


def web_link(paragraph, label, url):
    item = OxmlElement('w:hyperlink')
    item.set(qn('r:id'), paragraph.part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
    run = OxmlElement('w:r')
    prop = OxmlElement('w:rPr')
    color = OxmlElement('w:color')
    color.set(qn('w:val'), '1E536F')
    prop.append(color)
    underline = OxmlElement('w:u')
    underline.set(qn('w:val'), 'single')
    prop.append(underline)
    run.append(prop)
    content = OxmlElement('w:t')
    content.text = label
    run.append(content)
    item.append(run)
    paragraph._p.append(item)


def properties(parent, tag, values):
    node = OxmlElement(tag)
    for key, value in values.items():
        node.set(qn(f'w:{key}'), str(value))
    parent.append(node)
    return node


class Report:
    def __init__(self):
        self.doc = Document()
        self.section = self.doc.sections[0]
        self.section.page_width = Inches(8.5)
        self.section.page_height = Inches(11)
        self.section.top_margin = Inches(.72)
        self.section.bottom_margin = Inches(.66)
        self.section.left_margin = Inches(.79)
        self.section.right_margin = Inches(.79)
        self.section.header_distance = Inches(.29)
        self.section.footer_distance = Inches(.31)
        self.width = 6.92
        self._configure_styles()
        self._configure_pages()
        xsl = Path(getenv('OPENGCMC_MATHML_XSLT',
                          '/Applications/Microsoft Word.app/Contents/Resources/mathml2omml.xsl'))
        self.to_omml = etree.XSLT(etree.parse(str(xsl)))
        self.doc.core_properties.title = 'OpenGCMC 两阶段采样与双平台 CPU 实验报告'
        self.doc.core_properties.subject = 'CBMC 详平衡 NIST 参考复核 RASPA3 匹配对照与 CPU 性能'
        self.doc.core_properties.keywords = 'OpenGCMC, CBMC, GCMC, delayed acceptance, CPU'

    def _style(self, name, size, *, bold=False, east='Songti SC', color='000000',
               before=0, after=0, spacing=1.2, keep=False, latin='Arial'):
        styles = self.doc.styles
        if name not in styles:
            styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
        st = styles[name]
        font(st, size, east=east, latin=latin, bold=bold, color=color)
        st.paragraph_format.space_before = Pt(before)
        st.paragraph_format.space_after = Pt(after)
        st.paragraph_format.line_spacing = spacing
        st.paragraph_format.keep_with_next = keep
        st.paragraph_format.widow_control = True
        ppr = st.element.get_or_add_pPr()
        for border in ppr.findall(qn('w:pBdr')):
            ppr.remove(border)
        return st

    def _configure_styles(self):
        self._style('Normal', 11, after=8, spacing=Pt(17.4))
        self._style('Title', 21.5, east='Heiti SC', bold=True, after=9, spacing=Pt(30), keep=True)
        self._style('Subtitle', 10, east='Heiti SC', after=15, spacing=Pt(15), keep=True)
        self._style('Heading 1', 18, east='Heiti SC', bold=True, after=12, spacing=Pt(25), keep=True)
        self._style('Heading 2', 12.45, east='Heiti SC', bold=True, before=10, after=5,
                    spacing=Pt(18), keep=True)
        self._style('Table Text', 9.4, east='Heiti SC', spacing=1.15)
        self._style('Table Header', 9.35, east='Heiti SC', spacing=1.13, bold=True, color='FFFFFF')
        self._style('Table Title', 9.8, east='Heiti SC', bold=True, before=4, after=4, keep=True,
                    color=CAPTION_COLOR)
        self._style('Report Note', 9.35, color=CAPTION_COLOR, before=4, after=8, spacing=Pt(14.2))
        self._style('Figure Caption', 9.55, color=CAPTION_COLOR, before=3, after=8, spacing=Pt(14.5))
        self._style('Display Math', 11, before=3, after=7, spacing=1.12, latin='Cambria Math')
        self._style('Code Sample', 8.85, east='Heiti SC', latin='Menlo', before=4, after=9, spacing=1.24)
        self._style('Header', 8.4, east='Heiti SC', spacing=1.11)
        self._style('Footer', 8.4, east='Heiti SC', spacing=1.11)

    def _configure_pages(self):
        self.section.different_first_page_header_footer = True
        text(self.section.header.paragraphs[0], 'OpenGCMC　两阶段采样与双平台 CPU 基准')
        for footer in (self.section.footer, self.section.first_page_footer):
            p = footer.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            text(p, '第 ')
            field(p, ' PAGE ')
            text(p, ' 页')

    def para(self, value='', style=None, lead=None):
        p = self.doc.add_paragraph(style=style)
        if lead:
            text(p, lead, bold=True)
        text(p, value)
        return p

    def title(self, value, subtitle):
        self.para(value, 'Title')
        self.para(subtitle, 'Subtitle')

    def heading(self, value, level=2, new_page=False):
        p = self.para(value, f'Heading {level}')
        p.paragraph_format.page_break_before = new_page
        return p

    def math(self, markup):
        mathml = etree.fromstring(f'<math xmlns="{MML_NS}" display="block"><mrow>{markup}</mrow></math>')
        p = self.doc.add_paragraph(style='Display Math')
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p._p.append(self.to_omml(mathml).getroot())
        return p

    def table(self, caption, columns, rows, widths, left=(0,), note=None):
        if abs(sum(widths) - self.width) > .008:
            raise ValueError(f'Table {caption} must be {self.width} inches wide')
        self.para(caption, 'Table Title')
        table = self.doc.add_table(rows=1, cols=len(columns))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        for col, size in zip(table.columns, widths):
            col.width = Inches(size)
        tbl_pr = table._tbl.tblPr
        lines = properties(tbl_pr, 'w:tblBorders', {})
        for edge in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            properties(lines, f'w:{edge}', {'val': 'single', 'sz': 5, 'space': 0, 'color': 'D9D9D9'})
        marg = properties(tbl_pr, 'w:tblCellMar', {})
        for edge, size in (('top', 76), ('bottom', 76), ('left', 115), ('right', 115)):
            properties(marg, f'w:{edge}', {'w': size, 'type': 'dxa'})
        for i, values in enumerate((columns, *rows)):
            row = table.rows[0] if i == 0 else table.add_row()
            trpr = row._tr.get_or_add_trPr()
            properties(trpr, 'w:cantSplit', {})
            if i == 0:
                properties(trpr, 'w:tblHeader', {})
            for j, (cell, size, value) in enumerate(zip(row.cells, widths, values)):
                cell.width = Inches(size)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                fill = HEADER_FILL if i == 0 else 'F3F7FA' if i % 2 == 0 else 'FFFFFF'
                properties(cell._tc.get_or_add_tcPr(), 'w:shd', {'fill': fill, 'val': 'clear'})
                p = cell.paragraphs[0]
                p.style = 'Table Header' if i == 0 else 'Table Text'
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j in left else WD_ALIGN_PARAGRAPH.CENTER
                text(p, value)
        if note:
            self.para(note, 'Report Note')
        return table

    def figure(self, path, caption, width=6.82):
        p = self.doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_before = Pt(4)
        p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True
        drawing = p.add_run().add_picture(str(path), width=Inches(width))
        drawing._inline.docPr.set('descr', caption)
        self.para(caption, 'Figure Caption')

    def reference(self, citation, label, url):
        p = self.para(citation + '　', 'Report Note')
        web_link(p, label, url)


def mi(value, upright=False):
    flag = ' mathvariant="normal"' if upright else ''
    return f'<mi{flag}>{escape(str(value))}</mi>'


def mn(value):
    return f'<mn>{escape(str(value))}</mn>'


def mtext(value):
    return f'<mtext>{escape(str(value))}</mtext>'


def mo(value):
    return f'<mo>{escape(str(value))}</mo>'


def row(*parts):
    return '<mrow>' + ''.join(parts) + '</mrow>'


def sub(base, index):
    return '<msub>' + row(base) + row(index) + '</msub>'


def sup(base, index):
    return '<msup>' + row(base) + row(index) + '</msup>'


def frac(top, bottom):
    return '<mfrac>' + row(top) + row(bottom) + '</mfrac>'


def bracket(body, left='(', right=')'):
    return '<mfenced open="%s" close="%s">%s</mfenced>' % (escape(left), escape(right), row(body))


def radical(body):
    return '<msqrt>' + row(body) + '</msqrt>'


def under(base, lower):
    return '<munder>' + row(base) + row(lower) + '</munder>'


def accent(base, mark):
    return '<mover accent="true">' + row(base) + mo(mark) + '</mover>'
