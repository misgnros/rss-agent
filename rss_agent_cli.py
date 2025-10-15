import feedparser
import google.generativeai as genai
import sqlite3
import os
import asyncio
from typing import TypedDict
from contextlib import contextmanager
import re
from html import unescape
from urllib.parse import urlparse

from langgraph.graph import StateGraph, END
from typing_extensions import Annotated

# Import configuration
from config import (
    RSS_FEEDS,
    DB_FILE,
    AGENTS,
    GEMINI_MODEL,
    MAX_ARTICLES_PER_FEED,
    API_CALL_DELAY
)

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

@contextmanager
def get_db_connection():
    """Manage database connection with context manager"""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        yield conn
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize database"""
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE,
                    title TEXT,
                    summary TEXT,
                    full_content TEXT,
                    char_count INTEGER
                )
            """)
            conn.commit()
        print("Database initialization completed.")
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        raise

class ArticleState(TypedDict):
    url: str
    title: str
    summary: str
    full_content: str
    cleaned_content: str
    char_count: int
    keywords: list
    should_process: bool
    comments: dict

def clean_html(text: str) -> str:
    """Remove HTML tags and clean text"""
    if not text:
        return ""
    
    # Unescape HTML entities
    text = unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove multiple whitespaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_keywords(text: str, max_keywords: int = 5) -> list:
    """Extract simple keywords from text (basic implementation)"""
    if not text:
        return []
    
    # Remove common words (basic stopwords)
    stopwords = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    
    # Count frequency
    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:max_keywords]
    
    return [word for word, freq in keywords]

async def preprocess_article(state: ArticleState) -> dict:
    """Preprocess article: fetch full content, clean, and extract metadata"""
    print(f"\n[Preprocessing] {state['title']}")
    
    url = state['url']
    summary = state['summary']
    
    # Debug: Check original content length
    print(f"  Original content length: {len(summary)} characters")
    
    # Clean the summary text
    cleaned_summary = clean_html(summary)
    
    # Debug: Check cleaned content length
    print(f"  Cleaned content length: {len(cleaned_summary)} characters")
    
    # Try to fetch full content using web_fetch if available
    # For now, we'll use the summary as the content
    full_content = cleaned_summary
    
    # Calculate word count
    char_count = len(full_content.strip())
    
    # Extract keywords
    keywords = extract_keywords(full_content)
    
    # Determine if article should be processed
    # Skip if content is too short or URL is invalid
    should_process = True
    
    if char_count < 10:
        print(f"  ⚠ Article too short ({char_count} chars). Skipping.")
        print(f"  ⚠ First 200 chars of cleaned content: {cleaned_summary[:200]}")
        should_process = False
    
    if not url or not urlparse(url).scheme:
        print(f"  ⚠ Invalid URL. Skipping.")
        should_process = False
    
    if should_process:
        print(f"  ✓ Content cleaned: {char_count} chars")
        print(f"  ✓ Keywords: {', '.join(keywords)}")
    
    return {
        "full_content": full_content,
        "cleaned_content": cleaned_summary,
        "char_count": char_count,
        "keywords": keywords,
        "should_process": should_process
    }

async def generate_comment(state: ArticleState, agent_name: str, instruction: str) -> str:
    """Generate comment from AI agent"""
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            # Use cleaned content for better analysis
            content_preview = state['cleaned_content'][:500]  # Limit length for prompt
            
            prompt = f"""
あなたは{agent_name}です。
以下の記事に対して、{instruction}

記事タイトル: {state['title']}
キーワード: {', '.join(state.get('keywords', []))}
記事内容: {content_preview}
"""
            model = genai.GenerativeModel(GEMINI_MODEL)
            response = await model.generate_content_async(prompt)
            
            if not response or not response.text:
                raise ValueError("Empty response received")
            
            return response.text.strip()
            
        except Exception as e:
            print(f"Comment generation error ({agent_name}) - Attempt {attempt + 1}/{max_retries}: {e}")
            
            if attempt < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                return f"[Error: Failed to generate comment ({type(e).__name__})]"

async def process_article(state: ArticleState) -> dict:
    """Process article and generate comments"""
    
    # Check if article should be processed
    if not state.get('should_process', True):
        print(f"  → Skipping article due to preprocessing results")
        return {"comments": {}}
    
    article_url = state["url"]
    title = state["title"]
    summary = state["summary"]
    full_content = state.get("full_content", summary)
    char_count = state.get("char_count", 0)

    # Save article to database
    try:
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM articles WHERE url=?", (article_url,))
            
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO articles (url, title, summary, full_content, char_count) VALUES (?, ?, ?, ?, ?)",
                    (article_url, title, summary, full_content, char_count)
                )
                conn.commit()
                print(f"  ✓ Saved new article to database")
            else:
                print(f"  ℹ Article already exists in database")
    except sqlite3.Error as e:
        print(f"  ✗ Database save error: {e}")
    except Exception as e:
        print(f"  ✗ Unexpected error (database processing): {e}")

    # Generate comments
    comments = {}
    try:
        print(f"\n[Generating Comments]")
        for agent_name, instruction in AGENTS:
            print(f"  → {agent_name} is working...")
            comment = await generate_comment(state, agent_name, instruction)
            comments[agent_name] = comment
            await asyncio.sleep(API_CALL_DELAY)
        
        # Display comments
        print(f"Article: {title}")
        print(f"Chars: {char_count} | Keywords: {', '.join(state.get('keywords', []))}")
        for agent_name, comment in comments.items():
            print(f"\n[{agent_name}]")
            print(comment)
            print(f"{'-'*60}")
            
    except Exception as e:
        print(f"Error occurred during comment generation: {e}")

    return {"comments": comments}

def build_graph() -> StateGraph:
    """Build LangGraph workflow with two nodes"""
    graph = StateGraph(ArticleState)
    
    # Add nodes
    graph.add_node("preprocess_article", preprocess_article)
    graph.add_node("process_article", process_article)
    
    # Set entry point
    graph.set_entry_point("preprocess_article")
    
    # Add edges
    graph.add_edge("preprocess_article", "process_article")
    graph.add_edge("process_article", END)
    
    return graph

async def main():
    """Main processing"""
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        return
    
    # Initialize database
    try:
        init_db()
    except Exception as e:
        print(f"Program terminated due to database initialization failure: {e}")
        return

    # Process RSS feeds
    for feed_url in RSS_FEEDS:
        print(f"\n{'='*60}")
        print(f"Feed: {feed_url}")
        print(f"{'='*60}")
        
        try:
            feed = feedparser.parse(feed_url)
            
            # Check for parsing errors
            if hasattr(feed, 'bozo') and feed.bozo:
                print(f"Warning: Problem occurred while parsing RSS feed: {feed.bozo_exception}")
            
            if not feed.entries:
                print("No articles retrieved. Please check the feed URL.")
                continue
            
            print(f"{len(feed.entries)} articles found. Processing latest {MAX_ARTICLES_PER_FEED} articles.\n")
            
            # Process articles
            for idx, entry in enumerate(feed.entries[:MAX_ARTICLES_PER_FEED], 1):
                try:
                    article_url = str(entry.get('link', ''))
                    title = str(entry.get('title', '(No title)'))
                    
                    # Try to get content in order of preference:
                    # 1. content (full text in Atom feeds)
                    # 2. summary (RSS/Atom summary)
                    # 3. description (RSS description)
                    content_source = 'unknown'
                    summary = ''
                    
                    if hasattr(entry, 'content') and entry.content:
                        try:
                            summary = str(entry.content[0].get('value', ''))
                            if summary:
                                content_source = 'content (full text)'
                        except (IndexError, KeyError):
                            pass
                    
                    if not summary and hasattr(entry, 'summary'):
                        summary = str(entry.summary)
                        content_source = 'summary'
                    
                    if not summary and hasattr(entry, 'description'):
                        summary = str(entry.description)
                        content_source = 'description'
                    
                    if not summary:
                        summary = title
                        content_source = 'title only'
                    
                    if not article_url:
                        print(f"Article {idx}: Could not retrieve URL. Skipping.")
                        continue
                    
                    print(f"\n{'─'*60}")
                    print(f"Processing ({idx}/{min(MAX_ARTICLES_PER_FEED, len(feed.entries))}): {title}")
                    print(f"Content source: {content_source}")
                    print(f"{'─'*60}")
                    
                    state = ArticleState(
                        url=article_url,
                        title=title,
                        summary=summary,
                        full_content="",
                        cleaned_content="",
                        char_count=0,
                        keywords=[],
                        should_process=True,
                        comments={}
                    )

                    graph = build_graph().compile()
                    await graph.ainvoke(state)
                    
                except Exception as e:
                    print(f"Error occurred while processing article {idx}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error occurred while processing feed {feed_url}: {e}")
            continue
    
    print("\nAll processing completed.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nProgram interrupted.")
    except Exception as e:
        print(f"\nUnexpected error occurred: {e}")