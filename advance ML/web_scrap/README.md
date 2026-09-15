# Documentation Learning Website (Phase 1)

This project crawls technical documentation, cleans the content, and uses the Groq API (model: `groq/compound`) to generate beginner-friendly learning materials.

## Setup

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   playwright install chromium
   ```

2. Configure environment:
   Copy `.env.example` to `.env` and set your `GROQ_API_KEY`.

3. Run the application:
   ```bash
   streamlit run app.py
   ```
