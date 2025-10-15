# RSS feed URLs to monitor
RSS_FEEDS = [
    "https://zenn.dev/feed",
    "https://qiita.com/popular-items/feed.atom"
    # Add more RSS feeds here
]

# Database file path
DB_FILE = "rss_articles.db"

# AI agent configurations
# Format: (agent_name, instruction)
AGENTS = [
    ("肯定的なエージェント", "この記事の良い点や意義を前向きにコメントしてください。日本語100字程度で端的に。"),
    ("批判的なエージェント", "この記事の問題点や改善点を批判的にコメントしてください。日本語100字程度で端的に。"),
    ("中立的なエージェント", "この記事の内容を中立的に要約してください。日本語100字程度で端的に。")
]

# Gemini model to use
GEMINI_MODEL = "gemini-2.5-flash"

# Number of articles to process per feed
MAX_ARTICLES_PER_FEED = 1
# Delay between API calls (seconds)
API_CALL_DELAY = 1