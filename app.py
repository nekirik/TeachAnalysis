import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN


def get_stock_data(ticker, period='1y'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df['Close']


def normalize_pattern(pattern):
    """Normalize pattern to make it scale-invariant"""
    return (pattern - pattern.mean()) / pattern.std()


def find_similar_patterns(data, selected_point, pattern_length=30, projection_length=30, similarity_threshold=0.95,
                          search_range=None):
    # Get the pattern we're looking for (from selected point backwards)
    target_pattern = data[selected_point - pattern_length:selected_point]
    target_pattern_norm = normalize_pattern(target_pattern)
    patterns = []

    if search_range is None:
        search_range = (pattern_length, len(data) - pattern_length)

    # Find all similar patterns
    for i in range(search_range[0], search_range[1]):
        current_pattern = data[i - pattern_length:i]
        if len(current_pattern) == pattern_length:
            current_pattern_norm = normalize_pattern(current_pattern)

            # Calculate rolling correlation for better pattern matching
            correlation = np.correlate(
                current_pattern_norm - current_pattern_norm.mean(),
                target_pattern_norm - target_pattern_norm.mean(),
                mode='valid'
            )[0] / (pattern_length * current_pattern_norm.std() * target_pattern_norm.std())

            # Use both correlation and cosine similarity
            cosine_sim = cosine_similarity(
                current_pattern_norm.values.reshape(1, -1),
                target_pattern_norm.values.reshape(1, -1)
            )[0][0]

            # Combined similarity score
            similarity = (correlation + cosine_sim) / 2

            if similarity > similarity_threshold:
                # Get the continuation of the pattern
                continuation = data[i:i + projection_length]
                if len(continuation) >= projection_length:
                    # Scale and shift the continuation to match the selected point
                    scale_factor = data[selected_point] / current_pattern.iloc[-1]
                    adjusted_continuation = continuation * scale_factor

                    # Ensure the projection starts exactly from the selected point
                    shift = data[selected_point] - adjusted_continuation.iloc[0]
                    adjusted_continuation = adjusted_continuation + shift

                    patterns.append((i, current_pattern, adjusted_continuation, similarity))

    # Remove duplicates using clustering with adjusted parameters
    if patterns:
        # Extract continuations for clustering
        continuations = np.array([p[2].values for p in patterns])

        # Normalize continuations for clustering
        continuations_norm = (continuations - continuations.mean(axis=1, keepdims=True)) / continuations.std(axis=1,
                                                                                                             keepdims=True)

        # Perform DBSCAN clustering with adjusted parameters
        clustering = DBSCAN(eps=0.3, min_samples=1).fit(continuations_norm)

        # Keep the pattern with highest similarity from each cluster
        unique_patterns = []
        cluster_patterns = {}

        for idx, (i, pattern, continuation, similarity) in enumerate(patterns):
            cluster_label = clustering.labels_[idx]
            if cluster_label not in cluster_patterns or similarity > cluster_patterns[cluster_label][3]:
                cluster_patterns[cluster_label] = (i, pattern, continuation, similarity)

        unique_patterns = list(cluster_patterns.values())

        # Sort patterns by similarity
        unique_patterns.sort(key=lambda x: x[3], reverse=True)

        return unique_patterns

    return patterns


def plot_stock_data(data, selected_point, pattern_length, patterns=None):
    fig = go.Figure()

    # Plot main stock data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data.values,
        mode='lines',
        name='Actual Stock Price',
        line=dict(color='blue', width=2)
    ))

    # Highlight the selected pattern
    pattern_start = selected_point - pattern_length
    fig.add_trace(go.Scatter(
        x=data.index[pattern_start:selected_point],
        y=data.values[pattern_start:selected_point],
        mode='lines',
        name='Selected Pattern',
        line=dict(color='yellow', width=3)
    ))

    # Plot selected point
    fig.add_trace(go.Scatter(
        x=[data.index[selected_point]],
        y=[data.values[selected_point]],
        mode='markers',
        name='Selected Point',
        marker=dict(color='red', size=10)
    ))

    # Plot patterns and their projections
    if patterns:
        for i, pattern, continuation, similarity in patterns:
            # Plot the projection
            projection_dates = pd.date_range(
                start=data.index[selected_point],
                periods=len(continuation),
                freq=data.index.freq or 'B'
            )
            fig.add_trace(go.Scatter(
                x=projection_dates,
                y=continuation.values,
                mode='lines',
                line=dict(color='rgba(255,0,0,0.3)'),
                name=f'Projection (sim={similarity:.2f})',
                showlegend=True
            ))

    fig.update_layout(
        title='Stock Price Pattern Projections',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig


def main():
    st.title('Stock Pattern Projection Tool')

    # Input for stock ticker
    ticker = st.text_input('Enter Stock Ticker (e.g., AAPL):', 'AAPL')

    try:
        # Get stock data
        data = get_stock_data(ticker)

        if len(data) > 0:
            # Create two columns for controls
            col1, col2 = st.columns(2)

            with col1:
                # Add slider for pattern matching
                selected_point = st.slider(
                    'Select Point',
                    min_value=30,
                    max_value=len(data) - 1,
                    value=len(data) - 30
                )

                # Pattern length control
                pattern_length = st.slider(
                    'Pattern Length (days)',
                    min_value=5,
                    max_value=60,
                    value=30
                )

            with col2:
                # Add similarity threshold control
                similarity_threshold = st.slider(
                    'Similarity Threshold',
                    min_value=0.7,
                    max_value=1.0,
                    value=0.85,
                    step=0.01
                )

                # Projection length control
                projection_length = st.slider(
                    'Projection Length (days)',
                    min_value=5,
                    max_value=60,
                    value=30
                )

            # Find similar patterns
            patterns = find_similar_patterns(
                data,
                selected_point,
                pattern_length=pattern_length,
                projection_length=projection_length,
                similarity_threshold=similarity_threshold
            )

            # Create plot with patterns and projections
            fig = plot_stock_data(data, selected_point, pattern_length, patterns)
            st.plotly_chart(fig, use_container_width=True)

            # Display pattern information
            st.write(f'Found {len(patterns)} unique patterns')
            if patterns:
                st.write("Pattern Details:")
                for i, pattern, continuation, similarity in patterns:
                    st.write(
                        f"- Pattern found at {data.index[i].strftime('%Y-%m-%d')} with similarity {similarity:.2f}")

    except Exception as e:
        st.error(f'Error: {str(e)}')


if __name__ == '__main__':
    main() 