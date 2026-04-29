# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import re
import io
import os

# Optional AI integration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = FastAPI(title="Smart Resume/CV Optimizer v1.0")

# Allow local frontend (adjust origin for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResumeIn(BaseModel):
    text: str
    job_title: Optional[str] = None
    job_description: Optional[str] = None

class ResumeOut(BaseModel):
    optimized: str
    highlights: Dict[str, str]
    agent_scores: Dict[str, int]
    suggestions: List[str]

# --- Utilities: text extraction from files ---
def extract_text_from_pdf(fileobj) -> str:
    try:
        import PyPDF2
    except Exception:
        raise HTTPException(status_code=500, detail="PyPDF2 not installed")
    reader = PyPDF2.PdfReader(fileobj)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_text_from_docx(fileobj) -> str:
    try:
        import docx
    except Exception:
        raise HTTPException(status_code=500, detail="python-docx not installed")
    # python-docx expects a path or file-like; fileobj is file-like
    doc = docx.Document(fileobj)
    return "\n".join([p.text for p in doc.paragraphs])

# --- Core heuristics (your existing logic, extended) ---
def simple_optimize(text: str) -> str:
    # Trim, collapse multiple spaces, keep first two sentences per long line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    short_lines = []
    for line in lines:
        sentences = re.split(r'(?<=[.!?])\s+', line)
        if len(sentences) > 2:
            short = " ".join(sentences[:2])
        else:
            short = " ".join(sentences)
        short = re.sub(r'\s{2,}', ' ', short)
        short_lines.append(short)
    return "\n".join(short_lines)

def extract_highlights(text: str) -> Dict[str, str]:
    keywords = ["Python", "React", "FastAPI", "Django", "SQL", "JavaScript", "Tailwind", "AWS", "Docker", "Kubernetes"]
    found = [kw for kw in keywords if re.search(rf'\b{kw}\b', text, re.IGNORECASE)]
    # simple counts for quantified impact detection
    xyz_matches = re.findall(r'\b(?:\d{1,3}(?:,\d{3})*|\d+)%?\b', text)
    return {
        "skills": ", ".join(found) if found else "No clear skills found",
        "numbers_found": ", ".join(xyz_matches[:10]) if xyz_matches else "None"
    }

# --- Agent scoring heuristics ---
def recruiter_score(text: str, job_desc: Optional[str]) -> int:
    # ATS keyword matching: count overlap between job_desc keywords and resume
    if not job_desc:
        # fallback: count presence of common skills
        return min(30, len(re.findall(r'\b(Python|React|SQL|JavaScript|AWS|Docker)\b', text, re.IGNORECASE)) * 5)
    jd_keywords = re.findall(r'\b\w+\b', job_desc)
    jd_set = set([w.lower() for w in jd_keywords if len(w) > 3])
    resume_set = set([w.lower() for w in re.findall(r'\b\w+\b', text)])
    overlap = jd_set.intersection(resume_set)
    return min(40, len(overlap))

def hr_score(text: str) -> int:
    # F-pattern and social cues heuristic: presence of contact, LinkedIn, summary
    score = 0
    if re.search(r'linkedin\.com', text, re.IGNORECASE):
        score += 10
    if re.search(r'\bsummary\b|\bprofile\b', text, re.IGNORECASE):
        score += 10
    if re.search(r'\bteam\b|\bleadership\b|\bmanaged\b', text, re.IGNORECASE):
        score += 10
    return min(30, score)

def tech_score(text: str) -> int:
    # Technical depth: count technical keywords and quantified achievements
    tech_count = len(re.findall(r'\b(Python|Java|C\+\+|C#|React|FastAPI|Django|SQL|NoSQL|Kubernetes|Docker|AWS)\b', text, re.IGNORECASE))
    quantified = len(re.findall(r'\b(?:\d{1,3}(?:,\d{3})*|\d+)%?\b', text))
    return min(40, tech_count * 3 + quantified * 2)

def arbiter_score(scores: Dict[str,int]) -> int:
    # Simple weighted average
    total = scores.get("Recruiter",0)*0.35 + scores.get("HR",0)*0.2 + scores.get("Tech",0)*0.35 + scores.get("Arbiter",0)*0.1
    return int(round(total))

# --- Optional LLM helper (if OPENAI_API_KEY provided) ---
def call_llm_for_suggestions(text: str, job_title: Optional[str]=None) -> List[str]:
    if not OPENAI_API_KEY:
        # fallback heuristic suggestions
        suggestions = []
        if "Python" not in text:
            suggestions.append("If relevant, add Python experience with specific projects.")
        if "quantified" not in text.lower() and re.search(r'\b\d+\b', text):
            suggestions.append("Quantify achievements using numbers (X by Y using Z).")
        suggestions.append("Shorten long paragraphs into concise bullets.")
        return suggestions
    # If user has API key, call OpenAI (example)
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        prompt = f"Provide 4 concise resume improvement suggestions for this resume text. Job title: {job_title}\n\nResume:\n{text[:4000]}"
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200
        )
        out = resp.choices[0].message.content.strip()
        return [s.strip() for s in out.split("\n") if s.strip()]
    except Exception:
        return ["AI suggestion service failed; use heuristic suggestions instead."]

# --- Endpoints ---
@app.post("/optimize", response_model=ResumeOut)
async def optimize_resume(payload: ResumeIn):
    text = payload.text or ""
    optimized = simple_optimize(text)
    highlights = extract_highlights(text)
    # agent scores
    scores = {
        "Recruiter": recruiter_score(text, payload.job_description),
        "HR": hr_score(text),
        "Tech": tech_score(text),
    }
    # Arbiter initial placeholder (could be consensus)
    scores["Arbiter"] = int((scores["Recruiter"] + scores["HR"] + scores["Tech"]) / 3)
    final = arbiter_score(scores)
    suggestions = call_llm_for_suggestions(text, payload.job_title)
    return {
        "optimized": optimized,
        "highlights": highlights,
        "agent_scores": {**scores, "Final": final},
        "suggestions": suggestions
    }

@app.post("/optimize-file", response_model=ResumeOut)
async def optimize_resume_file(resume: UploadFile = File(...), job_title: Optional[str] = None, job_description: Optional[str] = None):
    filename = resume.filename.lower()
    content = ""
    # read file-like object from UploadFile
    if filename.endswith(".pdf"):
        content = extract_text_from_pdf(resume.file)
    elif filename.endswith(".docx"):
        # python-docx can read file-like objects
        content = extract_text_from_docx(resume.file)
    else:
        raw = await resume.read()
        try:
            content = raw.decode("utf-8")
        except Exception:
            content = raw.decode("latin-1", errors="ignore")
    # reuse optimize logic
    payload = ResumeIn(text=content, job_title=job_title, job_description=job_description)
    return await optimize_resume(payload)

@app.post("/mock-interview")
async def mock_interview(job_title: str, level: Optional[str] = "easy"):
    # Simple question generator based on job title
    base = job_title.lower()
    questions = []
    if "engineer" in base or "developer" in base or "software" in base:
        questions = [
            "Explain a recent technical challenge you solved and how you approached it.",
            "Describe a system you designed; what tradeoffs did you consider?",
            "How do you test and validate your code in production?"
        ]
    else:
        questions = [
            f"Why are you interested in the {job_title} role?",
            "Describe a time you handled a difficult stakeholder.",
            "What would you prioritize in your first 30 days?"
        ]
    if level == "easy":
        return {"questions": questions}
    # medium/hard could add coding tasks or case studies
    return {"questions": questions + ["Describe a complex scenario and how you'd measure success."]}

@app.post("/one-pager")
async def one_pager(job_title: str, resume_text: str):
    # Generate a 300-400 char intro; use LLM if available
    if OPENAI_API_KEY:
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            prompt = f"Write a 350-character professional one-paragraph introduction for a candidate applying to '{job_title}'. Use resume context: {resume_text[:1000]}"
            resp = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                max_tokens=200
            )
            return {"one_pager": resp.choices[0].message.content.strip()}
        except Exception:
            pass
    # fallback heuristic
    first_line = resume_text.splitlines()[0] if resume_text else ""
    one = f"Experienced professional applying for {job_title}. {first_line}"
    if len(one) > 400:
        one = one[:397] + "..."
    return {"one_pager": one}

@app.get("/")
def read_root():
    return {"message": "Smart Resume/CV Optimizer v1.0 backend is running"}
