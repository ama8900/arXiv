import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from pathlib import Path
import numpy as np

# Constants
DATA_DIR = "data"
CATEGORIES = [f.stem for f in Path(DATA_DIR).glob("*.csv")] if Path(DATA_DIR).exists() else []

@st.cache_data
def load_all_data():
    dfs = []
    for file in Path(DATA_DIR).glob("*.csv"):
        try:
            category = file.stem
            df = pd.read_csv(file)
            df['category'] = category
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {file.name}: {str(e)}")
            continue

    if not dfs:
        st.error("No valid CSV files found in the data directory!")
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)

# Data Loading and Preparation
full_df = load_all_data()

# Data Cleaning with Error Handling
if not full_df.empty:
    full_df['authors'] = full_df['authors'].fillna('').apply(lambda x: x.split(';') if isinstance(x, str) else [])
    full_df['subjects'] = full_df['subjects'].fillna('').apply(lambda x: x.split(',') if isinstance(x, str) else [])
    full_df['clean_title'] = full_df['title'].str.replace(r'[^a-zA-Z0-9 ]', '', regex=True).fillna('')

# UI Components
st.title("📊 ArXiv Interactive Dashboard")

if full_df.empty:
    st.warning("No data available to display!")
    st.stop()

# Category Comparison
st.header("📈 Category Comparison")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Selected Papers", len(full_df))
with col2:
    st.metric("Avg Papers per Category", round(len(full_df)/max(1, len(CATEGORIES)), 1))
with col3:
    top_author = full_df.explode('authors')['authors'].value_counts().index[0] if not full_df.empty else "N/A"
    st.metric("Top Author", top_author)

# Main Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Categories", "📚 Content", "🧑‍🤝‍🧑 Authors", "🔍 Papers"])

with tab1:
    st.subheader("📌 Category Distribution")
    if not full_df.empty:
        category_counts = full_df['category'].value_counts().reset_index()
        category_counts.columns = ['category', 'count']
        fig = px.bar(category_counts, x='count', y='category', orientation='h', color='category', height=600)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Subject Distribution Across Categories")
    try:
        exploded_subjects = full_df.explode('subjects')
        pivot_table = pd.pivot_table(
            exploded_subjects,
            index='subjects',
            columns='category',
            aggfunc='size',
            fill_value=0
        )
        heatmap = px.imshow(pivot_table.values,
                            labels=dict(x="Category", y="Subject", color="Count"),
                            x=pivot_table.columns,
                            y=pivot_table.index,
                            aspect="auto",
                            height=800)
        st.plotly_chart(heatmap, use_container_width=True)
    except Exception as e:
        st.error(f"Could not display subject distribution: {str(e)}")

with tab2:
    st.subheader("☁️ Word Clouds")
    if not full_df.empty:
        cols = st.columns(2)
        for idx, category in enumerate(CATEGORIES):
            text = ' '.join(full_df[full_df['category'] == category]['clean_title'].dropna())
            if text.strip():
                with cols[idx % 2]:
                    st.markdown(f"**{category}**")
                    try:
                        wordcloud = WordCloud(width=400, height=200, background_color='white').generate(text)
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Could not generate word cloud for {category}: {str(e)}")

with tab3:
    st.subheader("👥 Author Collaboration (Top 20)")
    category = st.selectbox("Select Category for Author Network", CATEGORIES)
    category_df = full_df[full_df['category'] == category]

    if not category_df.empty:
        try:
            authors = category_df.explode('authors')['authors'].value_counts().head(20).reset_index()
            authors.columns = ['Author', 'Count']
            fig = px.bar(authors, x='Count', y='Author', orientation='h', color='Count',
                         color_continuous_scale='Blues', height=600)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not display author network: {str(e)}")
    else:
        st.warning("No authors found for selected category")

with tab4:
    st.subheader("📝 Paper Explorer")
    category = st.selectbox("Select Category", CATEGORIES)
    category_papers = full_df[full_df['category'] == category]

    if not category_papers.empty:
        paper = st.selectbox(
            "Select Paper",
            options=category_papers['title'],
            format_func=lambda x: x[:70] + "..." if len(x) > 70 else x
        )
        selected = category_papers[category_papers['title'] == paper].iloc[0]

        st.markdown(f"### {selected['title']}")
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"**Category**: {selected['category']}")
            st.markdown(f"**Authors**: {', '.join(selected['authors']) if selected['authors'] else 'N/A'}")
            st.markdown(f"**Subjects**: {', '.join(map(str, selected['subjects'])) if selected['subjects'] else 'N/A'}")
        with col2:
            st.markdown(f"**Abstract**: [Link]({selected.get('abstract_url', '#')})")
            st.markdown(
                f"**[PDF]({selected.get('pdf_url', '#')}) | [HTML]({selected.get('html_url', '#')}) | [Other Formats]({selected.get('other_formats_url', '#')})**")
    else:
        st.warning("No papers in selected category")

# Summary Statistics
if not full_df.empty:
    try:
        st.subheader("📊 Summary Statistics")
        stats_df = full_df.groupby('category').agg(
            total_papers=('arxiv_id', 'count'),
            unique_authors=('authors', lambda x: x.explode().nunique()),
            unique_subjects=('subjects', lambda x: x.explode().nunique())
        ).reset_index()
        fig = go.Figure(data=[
            go.Bar(name='Total Papers', x=stats_df['category'], y=stats_df['total_papers']),
            go.Bar(name='Unique Authors', x=stats_df['category'], y=stats_df['unique_authors']),
            go.Bar(name='Unique Subjects', x=stats_df['category'], y=stats_df['unique_subjects'])
        ])
        fig.update_layout(barmode='group', xaxis_title="Category", yaxis_title="Count", height=600)
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Could not generate summary statistics: {str(e)}")
