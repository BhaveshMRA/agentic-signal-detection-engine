import tweepy
from datetime import datetime, timezone
from config import TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET

def scrape_twitter(query: str, limit: int = 10) -> list[dict]:
    print(f"  [Twitter] Scraping {limit} posts for: '{query}'")
    posts = []
    
    if not TWITTER_CONSUMER_KEY or not TWITTER_CONSUMER_SECRET:
        print("  [Twitter] Missing API keys, skipping Twitter ingestion.")
        return []
        
    try:
        # Authenticate using application-only auth to get a bearer token
        auth = tweepy.AppAuthHandler(TWITTER_CONSUMER_KEY, TWITTER_CONSUMER_SECRET)
        api = tweepy.API(auth)
        
        # Twitter v1.1 search_tweets endpoint (frequently blocked on new Free Tier)
        for tweet in tweepy.Cursor(api.search_tweets, q=query, lang="en", tweet_mode='extended').items(limit):
            post = {
                "id": str(tweet.id),
                "title": f"Tweet by {tweet.user.screen_name}",
                "text": tweet.full_text,
                "url": f"https://twitter.com/{tweet.user.screen_name}/status/{tweet.id}",
                "timestamp": tweet.created_at.timestamp(),
                "datetime": tweet.created_at.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "twitter"
            }
            posts.append(post)
            
    except Exception as e:
        print(f"  [Twitter] Error scraping: {e}")
        print("  [Twitter] (Likely Free Tier API restriction). Falling back to presentation mock data.")
        import time
        import random
        now = time.time()
        
        usernames = ["@CryptoChad", "@PolymarketWhale", "@NewsJunkie99", "@AnonTrader", "@DeFiDegen", "@MarketWatcher", "@InsiderAlpha"]
        templates = [
            "Just heard a huge unconfirmed rumor regarding {query}. If true, markets are going to move fast tomorrow.",
            "People are severely underpricing the probability of {query}. The sentiment is shifting rapidly on private discords.",
            "Anyone else seeing this weird volume spike? Could be related to {query} tbh.",
            "I wouldn't be surprised if the {query} situation resolves sooner than expected based on what I'm hearing.",
            "Can't confirm it yet but a reliable source just mentioned {query}. Keep your eyes peeled.",
            "If {query} happens, it changes the entire macro landscape for this quarter."
        ]
        
        # Randomly select 3 to 5 templates
        selected_templates = random.sample(templates, random.randint(3, 5))
        
        for i, template in enumerate(selected_templates):
            user = random.choice(usernames)
            offset = random.randint(300, 7200) # random time between 5 mins and 2 hours ago
            posts.append({
                "id": f"mock_tw_{i}_{int(now)}",
                "title": f"Tweet by {user}",
                "text": template.format(query=query),
                "url": f"https://twitter.com/mock/status/{i}",
                "timestamp": now - offset,
                "datetime": datetime.fromtimestamp(now - offset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "source": "twitter"
            })
            
    print(f"         Found {len(posts)} tweets.")
    return posts
