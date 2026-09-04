"""
Step 2: build a leakage-free feature table.

Design:
  - Observation window: days 0-13 after each user's install_date (14 days).
    All features are built ONLY from activity in this window.
  - Label window: days 14-27 after install (the next 14 days).
    churned = 1 if the user has NO session_start event in that window.
  - This keeps a hard wall between "what we know at prediction time" and
    "what we're trying to predict" -- the mistake the Cookie Cats dataset
    makes it easy to fall into (see that project's README/limitations).
"""
import pandas as pd

users = pd.read_excel("users.xlsx")
events = pd.read_excel("events.xlsx")

events = events.merge(users, on="user_id")
events["day_offset"] = (events["event_date"] - events["install_date"]).dt.days

FEATURE_WINDOW = range(0, 14)   # days 0-13
LABEL_WINDOW = range(14, 28)    # days 14-27

feat_events = events[events["day_offset"].isin(FEATURE_WINDOW)]
label_events = events[events["day_offset"].isin(LABEL_WINDOW)]

rows = []
for user_id, group in feat_events.groupby("user_id"):
    offsets = sorted(group["day_offset"].unique())
    days_active = len(offsets)
    last_active_day = max(offsets)
    recency = 13 - last_active_day  # days since last session, as of end of window
    first_week = sum(1 for d in offsets if d <= 6)
    second_week = sum(1 for d in offsets if d >= 7)
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    max_gap = max(gaps) if gaps else 0

    rows.append(
        {
            "user_id": user_id,
            "days_active_14d": days_active,
            "active_ratio_14d": round(days_active / 14, 4),
            "recency_days": recency,
            "first_week_sessions": first_week,
            "second_week_sessions": second_week,
            "trend": second_week - first_week,  # positive = engagement growing
            "max_gap_days": max_gap,
        }
    )

features = pd.DataFrame(rows)

# users with zero activity in the feature window can't have features computed
# from activity (days_active would be 0) -- but they DO still exist in `users`,
# so make sure every installed user is represented, filling zeros where a user
# had no session at all in the first 14 days.
all_users = users[["user_id"]].copy()
features = all_users.merge(features, on="user_id", how="left")
for col in ["days_active_14d", "active_ratio_14d", "first_week_sessions", "second_week_sessions", "trend"]:
    features[col] = features[col].fillna(0)
features["recency_days"] = features["recency_days"].fillna(13)  # never active = max recency
features["max_gap_days"] = features["max_gap_days"].fillna(0)

# label
churned_users = set(label_events["user_id"].unique())
features["churned"] = (~features["user_id"].isin(churned_users)).astype(int)

print(f"Total users: {len(features)}")
print(f"Churn rate (no activity in days 14-27): {features['churned'].mean():.4f}")
print()
print("Feature summary:")
print(features.describe().round(2))

features.to_csv("features.csv", index=False)
print("\nSaved features.csv")
