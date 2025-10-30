# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="NFL Player Prop Model", layout="centered")

st.title("🏈 NFL Player Prop Model")
st.write("Upload your weekly CSVs or connect Google Sheets to project player prop outcomes and probabilities.")

# --- Upload CSVs ---
st.header("📤 Upload Data Files")

team_csv = st.file_uploader("Upload Team Defense CSV", type=["csv"])
player_csv = st.file_uploader("Upload Player Game Log CSV", type=["csv"])

if not team_csv or not player_csv:
    st.info("⬆️ Please upload both CSVs to continue.")
    st.stop()

team_def = pd.read_csv(team_csv)
pgl = pd.read_csv(player_csv)
st.success("✅ Files uploaded successfully!")

# --- User Inputs ---
st.header("🧠 Select Player & Prop Type")

player = st.text_input("Enter Player Name (exact match from CSV)")
prop_type = st.selectbox("Prop Type", [
    "passing_yards", "rushing_yards", "receiving_yards",
    "receptions", "targets", "carries"
])
target_val = st.number_input("Target Value (e.g., 90, 5.5)", value=50.0)
opponent = st.text_input("Next Opponent (must match 'Team' in defense CSV)")

if not player or not opponent:
    st.warning("Please enter a player name and opponent to continue.")
    st.stop()

# --- Filter Player Data ---
pdf = pgl[pgl['Player'].str.lower() == player.lower()]
if pdf.empty:
    st.error("❌ Player not found in player_game_log CSV.")
    st.stop()

pos = pdf.iloc[0]['Position'].upper()

# --- Determine columns and defense mapping ---
aux = None
if prop_type == "passing_yards":
    stat_col = "Passing_Yards"
    def_col = "QB_Passing_Yards_Allowed_Per_Game"
    if "QB_Pass_Attempts_Allowed_Per_Game" in team_def.columns:
        aux = "QB_Pass_Attempts_Allowed_Per_Game"
elif prop_type == "rushing_yards":
    stat_col = "Rushing_Yards"
    def_col = "QB_Rushing_Yards_Allowed_Per_Game" if pos == "QB" else "RB_Rushing_Yards_Allowed_Per_Game"
elif prop_type == "receiving_yards":
    stat_col = "Receiving_Yards"
    if pos == "WR":
        def_col = "WR_Receiving_Yards_Allowed_Per_Game"
    elif pos == "TE":
        def_col = "TE_Receiving_Yards_Allowed_Per_Game"
    elif pos == "RB":
        def_col = "RB_Receiving_Yards_Allowed_Per_Game" if "RB_Receiving_Yards_Allowed_Per_Game" in team_def.columns else "WR_Receiving_Yards_Allowed_Per_Game"
    else:
        def_col = "WR_Receiving_Yards_Allowed_Per_Game"
elif prop_type in ["receptions", "targets"]:
    stat_col = "Receptions" if prop_type == "receptions" else "Targets"
    if pos == "WR":
        def_col = "WR_Receiving_Yards_Allowed_Per_Game"
    elif pos == "TE":
        def_col = "TE_Receiving_Yards_Allowed_Per_Game"
    elif pos == "RB":
        def_col = "RB_Receiving_Yards_Allowed_Per_Game" if "RB_Receiving_Yards_Allowed_Per_Game" in team_def.columns else "WR_Receiving_Yards_Allowed_Per_Game"
    else:
        def_col = "WR_Receiving_Yards_Allowed_Per_Game"
elif prop_type == "carries":
    stat_col = "Carries"
    def_col = "QB_Rushing_Yards_Allowed_Per_Game" if pos == "QB" else "RB_Rushing_Yards_Allowed_Per_Game"
else:
    st.error("Invalid prop type.")
    st.stop()

# --- Merge and Prepare Data ---
merged = pdf.merge(team_def, left_on="Opponent", right_on="Team", how="left")
merged["rolling_avg_3"] = merged[stat_col].rolling(3, 1).mean()
features = ["rolling_avg_3", def_col]
if aux:
    features.append(aux)

X = merged[features].fillna(0)
y = merged[stat_col]
if len(X) < 2:
    st.warning("⚠️ Not enough data to build a reliable model.")
    st.stop()

# --- Train Model ---
model = LinearRegression().fit(X, y)
preds = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, preds)) if len(y) > 1 else np.std(y)

# --- Prepare Next Opponent ---
opp = team_def[team_def["Team"].str.lower() == opponent.lower()]
if opp.empty:
    st.error("Opponent not found in defense CSV.")
    st.stop()

def_strength = opp.iloc[0][def_col]
roll_recent = pdf[stat_col].tail(3).mean()
feat_next = [roll_recent, def_strength]
if aux:
    feat_next.append(opp.iloc[0][aux])
pred_next = model.predict([feat_next])[0]

# --- Probability Calculation ---
z = (target_val - pred_next) / rmse if rmse > 0 else 1
prob_over = 1 - norm.cdf(z)
prob_under = 1 - prob_over

# --- Display Output ---
st.header("📊 Model Results")

st.markdown(f"""
**Player:** {player}  
**Target Value:** {target_val}  
**Average over last 3 {prop_type}:** {roll_recent:.2f}  
**Predicted stat:** {pred_next:.2f}  
**Probability of the over:** {prob_over*100:.1f}%  
**Probability of the under:** {prob_under*100:.1f}%
""")

# --- Charts ---
st.subheader("Predicted vs Target")
fig1, ax1 = plt.subplots()
ax1.bar(["Predicted", "Target"], [pred_next, target_val], color=["skyblue", "lightcoral"])
ax1.set_ylabel(prop_type)
st.pyplot(fig1)

st.subheader(f"{prop_type.capitalize()} vs Defense Metric (Trendline)")
fig2, ax2 = plt.subplots()
ax2.scatter(merged[def_col], merged[stat_col], color="gray")
xs = np.linspace(merged[def_col].min(), merged[def_col].max(), 100).reshape(-1, 1)
trend_model = LinearRegression().fit(merged[[def_col]].fillna(0), merged[stat_col])
ax2.plot(xs, trend_model.predict(xs), color="blue", label="Trendline")
ax2.set_xlabel(def_col)
ax2.set_ylabel(stat_col)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
