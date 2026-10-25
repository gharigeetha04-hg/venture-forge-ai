# Venture Forge AI

AI-powered platform that converts research papers and scientific ideas into structured startup opportunities.

## Getting a Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key

## Setup

### 1. Clone / Download the project

```bash
cd venture-forge-ai
```

### 2. Create your `.env` file

```bash
copy .env.example .env
```

Open `.env` and replace `your_gemini_api_key_here` with your actual key:

```
GEMINI_API_KEY=AIza...your_key_here
```

> **No key?** Leave the value empty or skip this step — the app runs in **Demo Mode** and returns a sample startup report.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Requires Python 3.9+

### 4. Run the app

```bash
python app.py
```

The app starts at **http://127.0.0.1:5000**

## Project Structure

```
venture-forge-ai/
  app.py              ← Flask backend (API + routes)
  requirements.txt    ← Python dependencies
  .env.example        ← Template for environment variables
  .env                ← Your actual secrets (git-ignored)
  database.db         ← Auto-created SQLite database
  templates/
    index.html        ← Landing page
    analyze.html      ← Upload / input page
    report.html       ← Generated startup report
    history.html      ← Past reports
  static/
    css/style.css     ← Dark theme styles
    js/main.js        ← Frontend logic
```

## Usage

1. Go to **Analyze** from the nav bar
2. Either drag-and-drop a research paper PDF **or** type your research idea
3. Click **"Generate Startup Opportunity"**
4. View your structured startup concept report
5. Copy, print/download, or revisit it later from **History**

## Disclaimer

AI-generated ideas should be independently validated for accuracy, market demand and revenue potential.
