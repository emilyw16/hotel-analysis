import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# Universal downloads folder
downloads = os.path.join(os.path.expanduser("~"), "Downloads")
csv_path = os.path.join(downloads, "nc_hotel_group.csv")

# Organize data
df = pd.read_csv(csv_path)
df = df[(df["advance_time"].notna()) & (df["advance_time"] >= 0) & (df["average_daily_rate"] > 0) & (df["average_daily_rate"] <= 400)].copy()
df["total_nights"] = df["weekend_stays"] + df["week_stays"]
df["booking_value"] = df["average_daily_rate"] * df["total_nights"]
df["advance_time_group"] = pd.cut(
    df["advance_time"], bins=[0, 7, 30, 90, 180, np.inf],
    labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"],
    include_lowest=True
)

# CHART 1
risk = (
    df.groupby(["property", "booking_channel", "advance_time_group"], observed=True)
    .agg(n_bookings=("booking_value", "size"),
         cancel_rate=("canceled", "mean"),
         avg_value=("booking_value", "mean"))
    .reset_index()
)
risk["dollars_at_risk"] = risk["cancel_rate"] * risk["avg_value"]
risk = risk[risk["n_bookings"] >= 10]

all_channels = sorted(risk["booking_channel"].unique())
properties = sorted(risk["property"].unique())
fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
vmax = risk["dollars_at_risk"].max()

for ax, prop in zip(axes, properties):
    sub = risk[risk["property"] == prop]
    pivot = sub.pivot(index="booking_channel", columns="advance_time_group", values="dollars_at_risk")
    pivot = pivot.reindex(index=all_channels,
                           columns=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181+ days"])
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="crest", vmin=0, vmax=vmax,
                linewidths=0.5, linecolor="white", ax=ax, cbar=(prop == properties[-1]),
                cbar_kws={"label": "Expected $ Lost per Booking"})
    ax.set_title(prop, fontsize=13)
    ax.set_xlabel("How Far Ahead the Booking Was Made")
    ax.set_ylabel("Booking Channel" if prop == properties[0] else "")

fig.suptitle(
    "Impact of Booking Channel and Advance Time on Expected Dollar Loss",
    fontsize=14, y=1.06
)
plt.tight_layout()
plt.show()

# CHART 2
weekly = (
    df.groupby(["property", "arrival_week_number"], observed=True)["canceled"]
    .agg(cancel_rate="mean", bookings="size")
    .reset_index()
)

plt.figure(figsize=(14, 6))
sns.lineplot(data=weekly, x="arrival_week_number", y="cancel_rate", hue="property", palette="crest", marker="o", markersize=4)
plt.title(
    "Which Weeks of the Year Have the Most Cancellations?\n"
    "Highest Peaks = Best Weeks to Overbook",
    fontsize=13
)
plt.xlabel("Arrival Week Number")
plt.ylabel("Cancellation Rate")
plt.legend(title="Hotel")
plt.tight_layout()
plt.show()

# CHART 3
rev = df.groupby(["property", "booking_channel"], observed=True).agg(
    cancelled_revenue=("booking_value", lambda x: x[df.loc[x.index, "canceled"] == 1].sum()),
    total_revenue=("booking_value", "sum"),
    n_bookings=("canceled", "size")
).reset_index()
rev["cancelled_revenue_ratio"] = (rev["cancelled_revenue"] / rev["total_revenue"]).round(3)
rev = rev[rev["n_bookings"] >= 10]

plt.figure(figsize=(10, 5.5))
sns.barplot(data=rev, x="booking_channel", y="cancelled_revenue_ratio", hue="property", palette="crest")
plt.title("Which Booking Channels Are More Reliable?\n"
    "Shorter Bars = More Reliable Channel",
    fontsize=13
)
plt.xlabel("Booking Channel")
plt.ylabel("Cancelled Revenue Ratio")
plt.xticks(rotation=25)
plt.legend(title="Hotel")
plt.tight_layout()
plt.show()