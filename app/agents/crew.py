"""
Mock multi-agent pipeline that mirrors a real CrewAI orchestration.

Each Agent has a role, goal, and a callable that receives RAG context and returns
a structured response. The Crew runs them sequentially, passing prior outputs as
context to downstream agents — the same pattern crewai.Process.sequential uses.

Swap `_mock_llm_call` for a real OpenAI/Anthropic call to go live with zero other changes.
"""
from dataclasses import dataclass, field
from app.rag.embedder import query as rag_query


# ---------------------------------------------------------------------------
# Minimal agent / crew primitives
# ---------------------------------------------------------------------------

@dataclass
class AgentOutput:
    raw: str


@dataclass
class Agent:
    role: str
    goal: str
    collections: list[str] = field(default_factory=list)

    def retrieve(self, query_text: str, n: int = 4) -> str:
        chunks = []
        for col in self.collections:
            chunks.extend(rag_query(query_text, col, n_results=n))
        return "\n---\n".join(chunks) if chunks else "(no context retrieved)"


@dataclass
class Task:
    description: str
    agent: Agent
    output: AgentOutput | None = None


class Crew:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def kickoff(self) -> list[AgentOutput]:
        context = ""
        for task in self.tasks:
            rag_context = task.agent.retrieve(task.description)
            raw = _mock_llm_call(task.agent.role, task.description, rag_context, context)
            task.output = AgentOutput(raw=raw)
            context += f"\n\n[{task.agent.role}]:\n{raw}"
        return [t.output for t in self.tasks]


# ---------------------------------------------------------------------------
# Mock LLM — replace this function to go live
# ---------------------------------------------------------------------------

_MOCK_RESPONSES: dict[str, str] = {
    "Resume Analyst": (
        "**Top Skills Identified:**\n"
        "• Python (5 years) — FastAPI, LangChain, CrewAI, scikit-learn, PyTorch\n"
        "• Cloud: AWS (EC2, S3, Lambda, SageMaker)\n"
        "• Vector Databases: ChromaDB, Pinecone, FAISS\n"
        "• SQL: PostgreSQL, query optimisation, schema design\n"
        "• Agentic AI: multi-agent orchestration, RAG pipelines, tool use\n\n"
        "**Strongest Achievements:**\n"
        "1. Built a RAG pipeline that reduced support ticket resolution time by 35%\n"
        "2. Fine-tuned an OpenAI model improving task accuracy by 22%\n"
        "3. Designed a recommendation model with 89% precision@5"
    ),
    "Job Description Analyst": (
        "**Must-Have Requirements:**\n"
        "• 2+ years Python in production\n"
        "• LLM application experience (RAG, agents, fine-tuning)\n"
        "• Vector database familiarity (ChromaDB / Pinecone / Weaviate)\n"
        "• REST API development (FastAPI preferred)\n"
        "• Cloud platform experience (AWS / GCP / Azure)\n\n"
        "**Nice-to-Have:**\n"
        "• Multi-agent frameworks (CrewAI, LangGraph, AutoGen)\n"
        "• LLM evaluation frameworks\n"
        "• MS/PhD in CS or AI"
    ),
    "Match Scorer": (
        '{\n'
        '  "score": 91,\n'
        '  "matched_skills": [\n'
        '    "Python (5 years, production)",\n'
        '    "FastAPI REST APIs",\n'
        '    "RAG pipeline (ChromaDB + LangChain)",\n'
        '    "AWS (EC2, S3, Lambda, SageMaker)",\n'
        '    "Multi-agent orchestration (CrewAI)",\n'
        '    "LLM fine-tuning",\n'
        '    "MS Computer Science"\n'
        '  ],\n'
        '  "gaps": [\n'
        '    "No explicit Docker / containerisation mentioned",\n'
        '    "No mention of LangGraph or AutoGen"\n'
        '  ]\n'
        '}'
    ),
    "Career Coach": (
        "**Recommendations (act on these this week):**\n\n"
        "1. **Add a Docker section to your resume.** The JD lists Docker in the tech stack "
        "and you have deployment experience — add one bullet per role showing you containerised "
        "a service (e.g., 'Dockerised FastAPI service, deployed to AWS Lambda via ECR').\n\n"
        "2. **Mention LangGraph or AutoGen by name.** You clearly know multi-agent patterns; "
        "recruiters keyword-search for specific framework names. Add a project note or skill tag.\n\n"
        "3. **Quantify your RAG project further.** '35% reduction' is great — add the scale "
        "(e.g., '10K docs indexed') and latency (e.g., 'p95 retrieval <200 ms') to make it "
        "concrete for a technical hiring panel.\n\n"
        "4. **Write a cover letter paragraph that mirrors their exact stack.** Use the phrase "
        "'production RAG pipeline' and list 'FastAPI + ChromaDB + AWS Lambda' — this project "
        "is a direct match and should be the lede.\n\n"
        "5. **Push this Career Copilot project to GitHub with a good README.** It demonstrates "
        "every skill on the JD in one repo and is a strong conversation starter in interviews."
    ),
}


def _mock_llm_call(role: str, task: str, rag_context: str, prior_context: str) -> str:
    """Return a realistic canned response keyed by agent role."""
    return _MOCK_RESPONSES.get(role, f"[{role}] No mock response defined for this role.")


# ---------------------------------------------------------------------------
# Public entry point (matches the real API contract)
# ---------------------------------------------------------------------------

def run_career_crew(job_title: str) -> dict:
    """Run the full mock crew pipeline and return structured results."""
    resume_agent = Agent("Resume Analyst", "Extract skills and achievements from the resume.", ["resumes"])
    job_agent = Agent("Job Description Analyst", "Extract requirements from the job description.", ["jobs"])
    scorer_agent = Agent("Match Scorer", "Score the resume against the job requirements.", ["resumes", "jobs"])
    coach_agent = Agent("Career Coach", "Give actionable recommendations to improve the application.", ["resumes", "jobs"])

    tasks = [
        Task(f"Analyse resume for candidate applying to '{job_title}'", resume_agent),
        Task(f"Analyse job description for '{job_title}'", job_agent),
        Task(f"Compute match score between resume and '{job_title}' JD", scorer_agent),
        Task(f"Give coaching recommendations for the '{job_title}' application", coach_agent),
    ]

    Crew(tasks).kickoff()

    return {
        "resume_analysis": tasks[0].output.raw,
        "job_analysis": tasks[1].output.raw,
        "match_score": tasks[2].output.raw,
        "recommendations": tasks[3].output.raw,
    }
