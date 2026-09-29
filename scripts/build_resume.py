"""Render Arishi Agarwal's CV to a one-page, ATS-friendly PDF.

Edit RESUME below and re-run; nothing about the layout needs touching.

    python -m scripts.build_resume [output.pdf]

Every number in the RetailOps entry is reproduced by a script in this repo:
  WAPE / interval coverage -> python -m forecasting.backtest
  guardrail coverage       -> pytest tests/unit/test_sql_guard.py
  transaction count        -> docs/data_cleaning.md
"""
from __future__ import annotations

import sys
from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import BaseDocTemplate, Flowable, Frame, ListFlowable, ListItem, PageTemplate, Paragraph, Spacer

INK = HexColor("#111111")
RULE = HexColor("#9a9a9a")

NAME = "Arishi Agarwal"
CONTACT = [
    "London, UK | +44 7352 669902 | arishiagarwal2002@gmail.com",
    "linkedin.com/in/arishiagarwal | github.com/arishiagarwal2002 | arishiagarwal.vercel.app",
    "Right to work: UK Graduate Route visa, no sponsorship required.",
]

SUMMARY = (
    "AI engineer building LLM and machine-learning systems in Python, from tool-using agents and "
    "retrieval-augmented generation to sequence models, with an emphasis on rigorous evaluation and "
    "safety guardrails. Backed by production engineering experience across REST APIs, relational schema "
    "design, automated testing and CI, and comfortable taking systems from design through to deployment."
)

SKILLS = [
    ("Languages", "Python, SQL, C++, JavaScript/TypeScript, Bash"),
    ("Machine Learning &amp; AI", "LLM Agents, Tool/Function Calling, Agent Evaluation, Retrieval-Augmented "
     "Generation, Deep Learning, NLP, Computer Vision, Time-Series Forecasting, Model Evaluation"),
    ("Frameworks", "PyTorch, TensorFlow, LangGraph, scikit-learn, LightGBM, Hugging Face, XGBoost, Pandas, "
     "NumPy, FAISS, pgvector"),
    ("Tools &amp; Infrastructure", "Git, Docker, CI/CD, FastAPI, Pydantic, PostgreSQL/MySQL, Alembic, PyTest, Linux"),
]

PROJECTS = [
    {
        "title": "RetailOps Agent: Tool-Using LLM Agent for Retail Operations",
        "stack": "Python, LangGraph, FastAPI, PostgreSQL/pgvector, LightGBM, sqlglot, Docker",
        "link": "github.com/arishiagarwal2002/retailops-agent",
        "bullets": [
            "Built a <b>tool-using LLM agent</b> that answers sales, inventory and returns questions over "
            "<b>1.02M real UK retail transactions</b> by chaining guarded text-to-SQL, demand forecasting and "
            "cited policy retrieval; proposed reorders are written as pending actions that require "
            "<b>human approval</b> before any state change.",
            "Engineered <b>defence-in-depth SQL guardrails</b> - sqlglot AST validation, a least-privilege "
            "read-only role and statement timeouts - blocking <b>all 20 adversarial write and escape patterns</b> "
            "under test (stacked statements, DML inside CTEs, system-catalog reads), with tests that bypass each "
            "layer to prove the others hold independently.",
            "Trained a <b>LightGBM quantile forecaster</b> (p10/p50/p90, direct multi-horizon) reaching "
            "<b>59.6% WAPE against 104.9% for a seasonal-naive baseline</b> on an 8-fold rolling-origin backtest "
            "(496 SKUs, 15,144 predictions), with <b>78.4% empirical coverage</b> of the nominal 80% prediction "
            "interval.",
            "Designed an <b>80-task benchmark with deterministic execution-match scoring</b>, per-category "
            "reporting and adversarial prompt-injection tasks, plus a deterministic verifier that checks every "
            "number in an answer against tool output; trace-level failure analysis showed five of six early "
            "failures were scorer defects rather than agent errors.",
        ],
    },
    {
        "title": "DevDocs RAG: Retrieval-Augmented Generation System",
        "stack": "Python, FastAPI, FAISS, BGE, LLM APIs, Docker",
        "link": "github.com/arishiagarwal2002/devdocs-rag",
        "bullets": [
            "Built a retrieval pipeline over 153 documents chunked into 2,208 passages, embedded into a FAISS "
            "index and served with grounded generation behind a containerised REST API.",
            "Wrote the evaluation harness before the features, scoring a hand-curated gold set on Recall@5, MRR "
            "and LLM-judged answer correctness; ablations showed BGE embeddings raised answer correctness from "
            "0.67 to 1.00, and cross-encoder re-ranking added 0.08 Recall@5 and 0.25 correctness.",
        ],
    },
    {
        "title": "Multivariate Time-Series Forecasting with Uncertainty",
        "stack": "Python, TensorFlow, Seq-to-Seq LSTM",
        "link": "MSc Dissertation",
        "bullets": [
            "Benchmarked missing-data imputation methods and a sequence-to-sequence LSTM forecaster against "
            "classical statistical baselines across a 420,551-record multivariate time series.",
            "Used Monte Carlo Dropout for calibrated prediction intervals instead of point estimates, with "
            "time-ordered training and validation splits.",
        ],
    },
    {
        "title": "Multimodal Transformer: Cross-Attention Image and Text Model",
        "stack": "PyTorch, ViT-B/16, BioBERT",
        "link": "",
        "bullets": [
            "Trained a 14-class multimodal classifier fusing a Vision Transformer image encoder and a BERT-family "
            "language encoder through cross-attention, using selective layer freezing for efficient fine-tuning "
            "under GPU memory constraints.",
        ],
    },
]

EXPERIENCE = [
    {
        "role": "Python Developer Intern", "org": "Suntist Labs", "dates": "Jan 2024 - Jun 2024",
        "bullets": [
            "Built and deployed 3 production Python modules with PyTest coverage, CI and monitoring, cutting "
            "release time by around 20%.",
            "Designed a REST-backed CRUD service over a normalised, indexed MySQL schema; query optimisation cut "
            "average retrieval time by 35% at 100 concurrent users.",
        ],
    },
    {
        "role": "Data Analyst Intern", "org": "MedTourEasy", "dates": "Jun 2023 - Dec 2023",
        "bullets": [
            "Analysed 50,000+ healthcare records in Python and SQL, raising data integrity by 15% and cutting "
            "manual-entry errors by 40% through automated validation.",
            "Built dashboards and reporting for senior stakeholders; findings informed three commercial decisions.",
        ],
    },
]

EDUCATION = [
    ("MSc Artificial Intelligence &amp; Machine Learning, Queen Mary University of London", "Sept 2025 - Sept 2026"),
    ("BTech Computer Science Engineering, Symbiosis Institute of Technology", "2020 - 2024"),
]

PUBLICATION = "SneakerXchange, International Conference on Innovative Computing and Communications (ICICC) 2026, Springer"
ACHIEVEMENTS = "3rd Place, N-Crypt N-Code | 2nd Place, Tinkerfest IoT"


class Rule(Flowable):
    """Thin horizontal rule under a section heading."""

    def __init__(self, width: float):
        super().__init__()
        self.width, self.height = width, 1.6

    def draw(self) -> None:
        self.canv.setStrokeColor(RULE)
        self.canv.setLineWidth(0.5)
        self.canv.line(0, 1, self.width, 1)


def styles(body_size: float, leading: float) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle("base", fontName="Helvetica", fontSize=body_size, leading=leading, textColor=INK)
    return {
        "name": ParagraphStyle("name", parent=base, fontName="Helvetica-Bold", fontSize=body_size + 8.5,
                               leading=body_size + 10, alignment=1, spaceAfter=2),
        "contact": ParagraphStyle("contact", parent=base, fontSize=body_size - 0.6,
                                  leading=body_size + 2.2, alignment=1),
        "section": ParagraphStyle("section", parent=base, fontName="Helvetica-Bold", fontSize=body_size + 1.2,
                                  leading=body_size + 3, spaceBefore=5.5, spaceAfter=1.5),
        "body": ParagraphStyle("body", parent=base, alignment=TA_JUSTIFY),
        "entry": ParagraphStyle("entry", parent=base, spaceBefore=3.2, spaceAfter=0.6),
        "bullet": ParagraphStyle("bullet", parent=base, alignment=TA_JUSTIFY, spaceAfter=1.2),
    }


def bullets(items: list[str], st: ParagraphStyle) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(t, st), leftIndent=9, value="bulletchar") for t in items],
        bulletType="bullet", start="•", leftIndent=9, bulletFontSize=st.fontSize,
        bulletOffsetY=-0.5, spaceBefore=0, spaceAfter=0,
    )


def build_story(s: dict, width: float) -> list:
    story: list = [Paragraph(NAME, s["name"])]
    story += [Paragraph(line, s["contact"]) for line in CONTACT]
    story.append(Spacer(1, 3))

    def section(title: str) -> None:
        story.append(Paragraph(title.upper(), s["section"]))
        story.append(Rule(width))

    section("Summary")
    story.append(Paragraph(SUMMARY, s["body"]))

    section("Technical Skills")
    for label, items in SKILLS:
        story.append(Paragraph(f"<b>{label}:</b> {items}", s["body"]))

    section("Projects")
    for p in PROJECTS:
        head = f'<b>{p["title"]}</b> | {p["stack"]}'
        if p["link"]:
            head += f'<br/><font size="{s["body"].fontSize - 0.8}">{p["link"]}</font>'
        story.append(Paragraph(head, s["entry"]))
        story.append(bullets(p["bullets"], s["bullet"]))

    section("Experience")
    for e in EXPERIENCE:
        story.append(Paragraph(f'<b>{e["role"]}</b>, {e["org"]}'
                               f'<font color="#555555"> &nbsp;|&nbsp; {e["dates"]}</font>', s["entry"]))
        story.append(bullets(e["bullets"], s["bullet"]))

    section("Education")
    for degree, dates in EDUCATION:
        story.append(Paragraph(f'{degree}<font color="#555555"> &nbsp;|&nbsp; {dates}</font>', s["body"]))

    section("Publication")
    story.append(Paragraph(PUBLICATION, s["body"]))

    section("Achievements")
    story.append(Paragraph(ACHIEVEMENTS, s["body"]))
    return story


def render(path: Path, body_size: float, leading: float, margin: float) -> int:
    doc = BaseDocTemplate(str(path), pagesize=A4, leftMargin=margin, rightMargin=margin,
                          topMargin=margin * 0.72, bottomMargin=margin * 0.6,
                          title=f"{NAME} - CV", author=NAME)
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="cv", frames=[frame])])
    doc.build(build_story(styles(body_size, leading), doc.width))
    return doc.page


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "Arishi_Agarwal_CV.pdf")
    # Shrink slightly until it fits one page, rather than cutting content.
    for size, leading, margin in ((9.3, 11.5, 15 * mm), (9.0, 11.1, 14 * mm),
                                  (8.7, 10.7, 13 * mm), (8.4, 10.3, 12 * mm)):
        if render(out, size, leading, margin) == 1:
            print(f"wrote {out} (1 page, {size}pt)")
            return
    print(f"wrote {out} (did not fit on one page; trim content in scripts/build_resume.py)")


if __name__ == "__main__":
    main()
