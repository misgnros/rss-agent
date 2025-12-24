# RSS feed URLs to monitor
RSS_FEEDS = [
    "https://zenn.dev/feed",
    # "https://qiita.com/popular-items/feed.atom"
    # Add more RSS feeds here
]

# Database file path
DB_FILE = "rss_articles.db"

# AI agent configurations
# Format: (agent_name, instruction)
AGENTS = [
    ("肯定的なエージェント", "この記事の良い点や意義を前向きにコメントしてください。"),
    ("批判的なエージェント", "この記事の問題点や改善点を批判的にコメントしてください。"),
    # ("中立的なエージェント", "この記事の内容を中立的に要約してください。")
]

# Gemini model to use
GEMINI_MODEL = "gemini-3-flash-preview"

# Number of articles to process per feed
MAX_ARTICLES_PER_FEED = 1
# Delay between API calls (seconds)
# NOTE: Free-tier Gemini API allows only ~5 requests/minute per model.
# Increase delay to stay under the rate limit when comments+rebuttals are enabled.
API_CALL_DELAY = 15