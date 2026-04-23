📰 NewsLens — Smart Personalized News Recommender

AI-powered hybrid news recommendation system using reinforcement learning, content intelligence, and contextual user signals.

🎯 Overview

NewsLens is a real-time personalized news recommendation system that adapts dynamically to user behavior.

Unlike traditional static recommenders, NewsLens continuously evolves using user interactions, contextual signals, and learning-based ranking strategies.

It delivers highly relevant news by combining machine learning, reinforcement learning, and NLP-based similarity models.

🧠 Core Idea

The system learns user preferences using multiple signals:

📖 Reading history (behavioral signal)
👥 Similar users (collaborative filtering)
📰 Article content (semantic similarity)
😊 User mood (context-aware personalization)
🔥 Trending/news popularity

All signals are fused into a unified ranking model.

⚖️ Ranking Strategy

Each article is assigned a final relevance score:

final_score =
    α * reinforcement_learning_score +
    β * content_similarity +
    γ * collaborative_signal +
    δ * mood_context +
    ε * popularity_score
🔄 Adaptive Weighting Strategy

Weights dynamically change based on user maturity:

User Stage	RL	Content	Collaborative	Mood	Popularity
Cold Start	0.10	0.10	0.00	0.50	0.30
New User	0.15	0.25	0.10	0.30	0.20
Active User	0.25	0.30	0.20	0.15	0.10
Mature User	0.35	0.25	0.25	0.10	0.05

👉 Early stage → more exploration + mood-based recommendations
👉 Later stage → more personalization + learning-based ranking

🧠 Recommendation Engine
1. Reinforcement Learning (Thompson Sampling)
Learns from user clicks and engagement
Balances exploration vs exploitation
Dynamically updates category preferences
2. Content-Based Filtering
TF-IDF vectorization of news articles
Cosine similarity with user history
Captures semantic relevance
3. Collaborative Filtering
Finds users with similar reading patterns
Recommends articles liked by similar users
4. Mood-Aware Personalization
Detects mood from user input / emojis
Adjusts category distribution dynamically
5. Popularity Signal
Boosts trending and widely read articles
Ensures freshness and relevance
📊 Dataset

We use the MIND (Microsoft News Dataset):

👥 ~50,000 users
📰 ~51,000 news articles
📈 ~150,000+ interactions
🏷️ 17 news categories

🔗 https://msnews.github.io/

🚀 Getting Started
1. Clone Repository
git clone https://github.com/amsheera2005/NewsLens.git
cd NewsLens
2. Create Virtual Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model
python train_model.py

This will:

Process MIND dataset
Build TF-IDF representations
Train reinforcement learning model
Save trained artifacts in /models
5. Run the App
streamlit run app.py

Open:

http://localhost:8501
🖥️ Features
📰 Personalized Feed
Smart ranking of articles
Continuously updated recommendations
😊 Mood-Based Adaptation
Mood input affects feed
Emotion-aware filtering
👍 Feedback Learning Loop
Like / Dislike signals
Real-time model updates
📊 Analytics Dashboard
User behavior insights
Category distribution
Recommendation performance
🧠 Self-Learning System
Improves with every interaction
No manual tuning required
📁 Project Structure
NewsLens/
│
├── app.py                      # Streamlit UI
├── data_processor.py          # Dataset preprocessing
├── recommendation_engine.py   # Hybrid recommender system
├── train_model.py             # Training pipeline
├── mood_handler.py            # Mood detection logic
├── database.py                # Interaction storage
├── styles.css                 # UI styling
├── requirements.txt           # Dependencies
├── README.md                  # Documentation
└── models/                    # Trained ML models
🔬 Personalization Signals

NewsLens adapts using:

Reading history
Click behavior
Mood input
Time of day
Category preference learning
Similar user patterns
⚡ Performance
⏱ Recommendation latency: < 2s
🧠 Training time: ~1 min (MIND subset)
💾 Lightweight ML pipeline (~200MB artifacts)
🏗️ System Architecture
User Input
   ↓
Streamlit Interface
   ↓
Hybrid Recommendation Engine
   ↓
(RL + Content + Collaborative + Mood + Popularity)
   ↓
Ranked Personalized News Feed
📚 References
Wu et al., MIND: A Large-scale Dataset for News Recommendation (ACL 2020)
Thompson Sampling (Multi-Armed Bandits)
TF-IDF + Cosine Similarity
Collaborative Filtering techniques
