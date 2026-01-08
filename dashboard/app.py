import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="FinBERT Sentiment Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom CSS for professional look
# -----------------------------
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #0E1117;
    }
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
    }
    
    /* Cards */
    .sentiment-card {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 5px solid;
    }
    
    .positive-card {
        background-color: #1a3d2e;
        border-color: #00C853;
    }
    
    .negative-card {
        background-color: #3d1a1a;
        border-color: #FF5252;
    }
    
    .neutral-card {
        background-color: #3d3a1a;
        border-color: #FFD700;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-weight: 600;
        padding: 12px;
        border-radius: 8px;
        border: none;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #1565C0;
        transform: translateY(-2px);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1a1f2e;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Initialize session state
# -----------------------------
if 'history' not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Sidebar - Model Info
# -----------------------------
st.sidebar.title("🔧 Model Information")
st.sidebar.markdown("""
**Model:** FinBERT  
**Framework:** PyTorch  
**Acceleration:** GPU (RTX 4050)  
**Dataset:** Financial PhraseBank  
**Task:** Sentiment Classification  
**Classes:** Positive, Neutral, Negative
""")

st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "📍 Select Analysis Mode",
    ["Manual Text", "Live Financial News"]
)

st.sidebar.markdown("---")
st.sidebar.title("📊 Analysis Stats")
if st.session_state.history:
    total = len(st.session_state.history)
    pos = sum(1 for h in st.session_state.history if h['sentiment'].lower() == 'positive')
    neg = sum(1 for h in st.session_state.history if h['sentiment'].lower() == 'negative')
    neu = total - pos - neg
    
    st.sidebar.metric("Total Analyses", total)
    st.sidebar.metric("Positive", f"{pos} ({pos/total*100:.0f}%)")
    st.sidebar.metric("Negative", f"{neg} ({neg/total*100:.0f}%)")
    st.sidebar.metric("Neutral", f"{neu} ({neu/total*100:.0f}%)")
else:
    st.sidebar.info("No analyses yet")

# -----------------------------
# Main header
# -----------------------------
st.title("📊 FinBERT Sentiment Analysis Dashboard")
st.markdown("**Professional-grade financial sentiment analysis powered by GPU-accelerated FinBERT**")
st.markdown("---")

# -----------------------------
# API config
# -----------------------------
API_URL = "http://127.0.0.1:8000/predict"
NEWS_API_URL = "http://127.0.0.1:8000/news"

# -----------------------------
# Main layout
# -----------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📝 Text Input")
    
    if mode == "Manual Text":
        text = st.text_area(
            "Enter financial text to analyze",
            value="The company reported strong quarterly profits and optimistic future growth.",
            height=150,
            help="Enter any financial news, earnings report, or market commentary"
        )
    else:
        st.info("📰 Live news mode selected. Click 'Analyze Sentiment' to fetch latest financial news.")
        text = ""  # Empty for live news mode
    
    analyze_btn = st.button("🚀 Analyze Sentiment", use_container_width=True)

with col2:
    st.subheader("⚡ Quick Actions")
    if st.button("📋 View History", use_container_width=True):
        st.session_state.show_history = True
    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    if st.button("📥 Download CSV", use_container_width=True):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download",
                csv,
                "sentiment_history.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.warning("No data to download")

# -----------------------------
# Analysis logic
# -----------------------------
if analyze_btn:
    if mode == "Manual Text":
        if not text.strip():
            st.warning("⚠️ Please enter some text to analyze")
        else:
            with st.spinner("🔄 Analyzing sentiment..."):
                try:
                    response = requests.post(
                        API_URL,
                        json={"text": text},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        label = result["label"]
                        confidence = result["confidence"]
                        
                        # Assuming API returns all probabilities
                        probs = result.get("probabilities", {
                            "positive": 0.0,
                            "neutral": 0.0,
                            "negative": 0.0
                        })
                        
                        # Add to history
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "text": text[:50] + "..." if len(text) > 50 else text,
                            "sentiment": label,
                            "confidence": confidence
                        })
                        
                        st.success("✅ Analysis Complete!")
                        
                        # Results display
                        st.markdown("---")
                        st.subheader("📊 Results")
                        
                        # Sentiment card with color
                        sentiment_lower = label.lower()
                        if sentiment_lower == "positive":
                            card_class = "positive-card"
                            emoji = "📈"
                            color = "#00C853"
                            message = "Positive market signal detected"
                        elif sentiment_lower == "negative":
                            card_class = "negative-card"
                            emoji = "📉"
                            color = "#FF5252"
                            message = "Negative market signal detected"
                        else:
                            card_class = "neutral-card"
                            emoji = "⚖️"
                            color = "#FFD700"
                            message = "Neutral sentiment detected"
                        
                        st.markdown(f"""
                        <div class="sentiment-card {card_class}">
                            <h2>{emoji} {label.upper()}</h2>
                            <p style="font-size: 18px; margin-top: 10px;">{message}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Metrics row
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Primary Sentiment", label.upper())
                        with col2:
                            st.metric("Confidence Score", f"{confidence:.1%}")
                        with col3:
                            st.metric("Certainty", "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low")
                        
                        # Confidence bar
                        st.markdown("**Confidence Level**")
                        st.progress(confidence)
                        
                        # Probability breakdown
                        st.markdown("---")
                        st.subheader("📊 Detailed Probability Breakdown")
                        
                        prob_col1, prob_col2, prob_col3 = st.columns(3)
                        
                        with prob_col1:
                            pos_prob = probs.get("positive", 0.0)
                            st.markdown(f"""
                            <div style="background-color: #1a3d2e; padding: 15px; border-radius: 8px; border-left: 4px solid #00C853;">
                                <h4 style="color: #00C853; margin: 0;">📈 Positive</h4>
                                <h2 style="margin: 10px 0;">{pos_prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(pos_prob)
                        
                        with prob_col2:
                            neu_prob = probs.get("neutral", 0.0)
                            st.markdown(f"""
                            <div style="background-color: #3d3a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #FFD700;">
                                <h4 style="color: #FFD700; margin: 0;">⚖️ Neutral</h4>
                                <h2 style="margin: 10px 0;">{neu_prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(neu_prob)
                        
                        with prob_col3:
                            neg_prob = probs.get("negative", 0.0)
                            st.markdown(f"""
                            <div style="background-color: #3d1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #FF5252;">
                                <h4 style="color: #FF5252; margin: 0;">📉 Negative</h4>
                                <h2 style="margin: 10px 0;">{neg_prob:.1%}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            st.progress(neg_prob)

                    else:
                        st.error(f"❌ API Error: {response.text}")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Request timed out. Please try again.")
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Cannot connect to API. Make sure the server is running on http://127.0.0.1:8000")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    elif mode == "Live Financial News":
        with st.spinner("🔄 Fetching live financial news..."):
            try:
                response = requests.get(
                    NEWS_API_URL,
                    params={"query": "stock", "limit": 5},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    
                    st.success("✅ Live News Sentiment Analysis Complete")
                    st.markdown("---")
                    st.subheader("📰 Latest Financial News Sentiment")
                    
                    for idx, item in enumerate(data["results"], 1):
                        sentiment_lower = item['sentiment'].lower()
                        
                        if sentiment_lower == "positive":
                            card_class = "positive-card"
                            emoji = "📈"
                        elif sentiment_lower == "negative":
                            card_class = "negative-card"
                            emoji = "📉"
                        else:
                            card_class = "neutral-card"
                            emoji = "⚖️"
                        
                        st.markdown(f"""
                        <div class="sentiment-card {card_class}">
                            <h3>{emoji} News #{idx}: {item['title']}</h3>
                            <p><strong>Sentiment:</strong> {item['sentiment'].upper()} | <strong>Confidence:</strong> {item['confidence']:.1%}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add to history
                        st.session_state.history.append({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "text": item['title'][:50] + "..." if len(item['title']) > 50 else item['title'],
                            "sentiment": item['sentiment'],
                            "confidence": item['confidence']
                        })
                    
                else:
                    st.error(f"❌ Failed to fetch news: {response.text}")

            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out. Please try again.")
            except requests.exceptions.ConnectionError:
                st.error("🔌 Cannot connect to API. Make sure the server is running on http://127.0.0.1:8000")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

# -----------------------------
# History section
# -----------------------------
if st.session_state.history:
    st.markdown("---")
    st.subheader("📜 Analysis History")
    
    # Create DataFrame
    df_history = pd.DataFrame(st.session_state.history)
    
    # Display table
    st.dataframe(
        df_history,
        use_container_width=True,
        hide_index=True
    )
    
    # Trend chart
    if len(st.session_state.history) > 1:
        st.markdown("---")
        st.subheader("📈 Sentiment Trend")
        
        # Prepare data for chart
        sentiment_map = {"positive": 1, "neutral": 0, "negative": -1}
        df_history['sentiment_value'] = df_history['sentiment'].str.lower().map(sentiment_map)
        df_history['color'] = df_history['sentiment'].str.lower().map({
            "positive": "#00C853",
            "neutral": "#FFD700",
            "negative": "#FF5252"
        })
        
        # Create plotly chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(df_history))),
            y=df_history['confidence'],
            mode='lines+markers',
            name='Confidence',
            line=dict(color='#1E88E5', width=3),
            marker=dict(
                size=12,
                color=df_history['color'],
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Analysis #%{x}</b><br>Confidence: %{y:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            plot_bgcolor='#0E1117',
            paper_bgcolor='#0E1117',
            font=dict(color='white'),
            xaxis=dict(
                title='Analysis Number',
                gridcolor='#2b3035',
                showgrid=True
            ),
            yaxis=dict(
                title='Confidence Score',
                gridcolor='#2b3035',
                showgrid=True,
                tickformat='.0%'
            ),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Built with FinBERT • GPU Accelerated • Real-time Analysis</p>
</div>
""", unsafe_allow_html=True)