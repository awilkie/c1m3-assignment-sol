# C1M3 Assignment Walkthrough

We have successfully implemented the solution for the C1M3 Assignment, integrating Google Gemini models and replacing the Tavily search tool with DuckDuckGo.

## Changes Implemented

### 1. Environment and Dependencies
- Created a Python virtual environment (`.venv`).
- Installed necessary dependencies: `openai`, `python-dotenv`, `duckduckgo-search`, `google-cloud-aiplatform` (for aisuite attempt, though ultimately unused), `requests`.
- **Note**: Switched to using `duckduckgo-search` instead of `tavily-python` to avoid API key requirements.

### 2. Code Implementation (`C1M3_Assignment.py`)
- **Client Initialization**: Configured `OpenAI` client to use Google's OpenAI-compatible endpoint (`https://generativelanguage.googleapis.com/v1beta/openai/`) with `GOOGLE_API_KEY`.
- **Model Selection**: 
    - Updated function signatures to default to `gemini-2.0-flash-exp` to ensure compatibility with the Google endpoint.
    - Refactored functions to explicitly use the `model` argument, allowing flexibility if the client backend changes.
    - Note: The original assignment intended `gpt-4o`, but `gemini-2.0-flash-exp` is used here for the Google setup.
- **Tool Integration**:
    - Replaced `tavily_search_tool` with `web_search_tool` (powered by DuckDuckGo) in `research_tools.py` and `C1M3_Assignment.py`.
    - Implemented `generate_research_report_with_tools` to handle tool calling loop.
- **Reflection**: Implemented `reflection_and_rewrite` with robust JSON parsing to handle markdown code blocks often returned by the model.
- **HTML Conversion**: Implemented `convert_report_to_html` to transform the report into a styled HTML document.

### 3. Verification
- Ran `C1M3_Assignment.py` which executes the full pipeline and runs unit tests.
- All tests passed, including:
    - Tool calling (Arxiv + DuckDuckGo)
    - Research report generation
    - Reflection and rewriting (with keyword checks)
    - HTML conversion

### 4. GitHub Publication
- Created a public repository: `https://github.com/awilkie/c1m3-assignment-sol`
- Pushed the solution code (excluding sensitive `.env` file).

## How to Run

1.  **Activate Virtual Environment**:
    ```bash
    source .venv/bin/activate
    ```
2.  **Run the script**:
    ```bash
    python C1M3_Assignment.py
    ```
    This will generate the research report, reflection, and HTML output, and run the included unit tests.

## Artifacts
- [C1M3_Assignment.py](file:///home/andrew/Documents/training/deeplearning.ai/agentic-ai/C1M3_Assignment/C1M3_Assignment.py)
- [research_tools.py](file:///home/andrew/Documents/training/deeplearning.ai/agentic-ai/C1M3_Assignment/research_tools.py)
