# Agentic AI QA Automation Framework

**Project Status:** Active Development
**Test Status:** 57 Tests Passing

## Overview

The Agentic AI QA Automation Framework is a Python-based project that applies Large Language Models (LLMs), Retrieval-Augmented Generation (RAG), and multi-agent orchestration to software quality assurance.

The framework analyzes software requirements, identifies potential quality risks, and automatically generates structured test plans using specialized AI agents.

The project demonstrates the integration of traditional Software Quality Assurance engineering practices with modern AI Engineering concepts.

## Key Features

- **AI-Powered Requirements Analysis:** Analyzes software requirements and identifies ambiguities, missing information, and potential quality risks.
- **Automated Test Planning:** Generates structured test scenarios, including positive, negative, and edge cases.
- **Agent-Based Architecture:** Separates requirements analysis and test planning into specialized AI agents.
- **RAG Integration:** Uses contextual information retrieved from a vector database to improve AI-generated analysis.
- **Vector Database:** Stores and retrieves relevant knowledge using ChromaDB.
- **Embedding Generation:** Uses OpenAI embedding models to generate vector representations of text.
- **Workflow Orchestration:** Implements sequential agent execution using LangGraph.
- **Dependency Injection:** Supports injected agents and services for easier testing and extensibility.
- **Automated Testing:** Includes unit and integration tests using pytest.

## Architecture

```text
05-agentic-ai-qa-framework/
│
├── src/
│   ├── agents/
│   │   ├── requirements_analyzer.py
│   │   └── planner_agent.py
│   │
│   ├── core/
│   │   ├── qa_orchestrator.py
│   │   └── qa_workflow.py
│   │
│   └── retrieval/
│       ├── embedding_service.py
│       └── knowledge_base.py
│
├── tests/
│   ├── integration/
│   │   └── test_qa_rag_integration.py
│   │
│   └── unit/
│       ├── test_embedding_service.py
│       ├── test_knowledge_base.py
│       ├── test_qa_orchestrator.py
│       ├── test_qa_workflow.py
│       ├── test_requirements_analyzer.py
│       └── test_test_planner.py
│
├── .env.example
├── .gitignore
├── README.md
└── requirements.txt
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python 3.12 | Core development language |
| OpenAI API | LLM and embedding integration |
| LangGraph | Agent workflow orchestration |
| ChromaDB | Vector database |
| RAG | Context retrieval and augmentation |
| pytest | Unit and integration testing |
| python-dotenv | Environment configuration |

## How It Works

The framework follows a sequential AI-driven workflow:

1. **Requirement Input:** Receives a software requirement.
2. **Knowledge Retrieval:** Searches the vector database for relevant contextual information when a knowledge base is configured.
3. **Requirements Analysis:** The Requirements Analyzer evaluates the requirement and identifies ambiguities, missing information, quality risks, acceptance criteria, and suggested test scenarios.
4. **Test Planning:** The Test Planner transforms the analysis into a structured test plan.
5. **Output:** Returns the requirements analysis and generated test plan.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bailona/05-agentic-ai-qa-framework.git

cd 05-agentic-ai-qa-framework
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file based on `.env.example`:

```powershell
Copy-Item .env.example .env
```

Add your OpenAI API key to `.env`:

```dotenv
OPENAI_API_KEY=your_openai_api_key_here
```

**Security:** Never commit your `.env` file or expose your API key.

## Usage

The framework can be used through the QA orchestrator.

Example:

```python
from src.core.qa_orchestrator import QAOrchestrator

orchestrator = QAOrchestrator()

result = orchestrator.process_requirement(
    "Users must be able to log in using their email and password."
)

print("Requirements Analysis:")
print(result["analysis"])

print("\nGenerated Test Plan:")
print(result["test_plan"])
```

The example requires a valid OpenAI API key and API access.

## Running Tests

Run the complete test suite:

```bash
python -m pytest -v
```

### Current Test Results

| Test Category | Result |
|---|---:|
| Unit Tests | 53 passed |
| Integration Tests | 4 passed |
| **Total** | **57 passed** |

Integration tests use mocks for external AI services where appropriate, avoiding unnecessary API calls and costs.

## Security and Best Practices

- API keys are stored in environment variables.
- `.env` is excluded through `.gitignore`.
- Virtual environments and local database files are excluded from version control.
- Dependencies are pinned to specific versions.
- Automated tests use dependency injection and mocks to isolate external services.

## Roadmap

The following capabilities are planned for future development:

- Automated API testing integration.
- UI test automation using Playwright.
- AI-assisted test execution and failure analysis.
- Automated test reporting.
- Requirements-to-test traceability.
- CI/CD pipeline integration.
- Expanded multi-agent collaboration.

These items are planned enhancements and are not represented as completed features.

## Author

**Paulo Anderson Bailona**

Senior QA Automation Engineer | SDET | AI Engineering

- GitHub: https://github.com/bailona
- LinkedIn: https://www.linkedin.com/in/paulo-anderson-b-174127136/
