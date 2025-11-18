import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from apputil import update_board

st.set_page_config(page_title="Conway's Game of Life", layout="centered")

st.title("Conway's Game of Life")

# Sidebar controls 
st.sidebar.header("Controls")

board_size = st.sidebar.slider(
    "Board Size (N x N)", min_value=5, max_value=50, value=10
)

run_steps = st.sidebar.number_input(
    "Number of auto-run steps", min_value=1, max_value=200, value=10
)

# Session State Setup
if "board" not in st.session_state:
    st.session_state.board = np.random.randint(2, size=(board_size, board_size))

# If board size changes to reset board
if st.session_state.board.shape[0] != board_size:
    st.session_state.board = np.random.randint(2, size=(board_size, board_size))

# Buttons
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Random Board"):
        st.session_state.board = np.random.randint(2, size=(board_size, board_size))

with col2:
    if st.button("Next Step"):
        st.session_state.board = update_board(st.session_state.board)

with col3:
    if st.button(" Auto Run"):
        for _ in range(run_steps):
            st.session_state.board = update_board(st.session_state.board)

# Display board 
st.subheader("Current Board")

fig, ax = plt.subplots()
sns.heatmap(
    st.session_state.board,
    cmap="plasma",
    cbar=False,
    square=True,
    xticklabels=False,
    yticklabels=False,
    ax=ax
)
st.pyplot(fig)