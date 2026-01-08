import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.ensemble import RandomForestClassifier


### 33 Dataset loading function
def load_datasetes():
    right = pd.read_csv("right_data.csv")
    left = pd.read_csv("left_data.csv")


    right.columns = right.columns.str.strip()
    left.columns = left.columns.str.strip()


    right["label"] = "Right"
    left["label"] = "Left"


    data = pd.concat([right, left], ignore_index=True)
    return left, right, data



# ##Magnitude computation

def compute_magnitude(df, cols):
    x = df[cols[0]].astype(float).to_numpy()
    y = df[cols[1]].astype(float).to_numpy()
    z = df[cols[2]].astype(float).to_numpy()
    return np.sqrt(x**2 + y**2 + z**2)


# ##Magnitude vs sample index

def plot_timeseries(l_df, r_df, cols, title):
    l_mag = compute_magnitude(l_df, cols)
    r_mag = compute_magnitude(r_df, cols)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(r_mag, label="Right")
    ax.plot(l_mag, label="Left")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Magnitude")
    ax.legend()
    ax.grid(True)

    fig.tight_layout()
    fig.savefig(title.replace(" ", "_"), dpi=300)
    plt.close(fig)



# ##Single-axis time series plots

def plot_single_axis_timeseries(l_df, r_df, cols, title):

    fig = plt.figure(figsize=(10, 6))
    gs = GridSpec(3, 1, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[2, 0])

    ax1.plot(r_df[cols[0]], label="Right")
    ax1.plot(l_df[cols[0]], label="Left")
    ax1.set_ylabel(f"{title[:3].lower()}_x")
    ax1.grid(True)

    ax2.plot(r_df[cols[1]])
    ax2.plot(l_df[cols[1]])
    ax2.set_ylabel(f"{title[:3].lower()}_y")
    ax2.grid(True)

    ax3.plot(r_df[cols[2]])
    ax3.plot(l_df[cols[2]])
    ax3.set_ylabel(f"{title[:3].lower()}_z")
    ax3.set_xlabel("Sample index")
    ax3.grid(True)

    fig.suptitle(title)
    fig.legend()
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f"Single_Axis_{title.split()[0]}", dpi=300)
    plt.close(fig)


# ##Pairplot

def plot_pairplot(data, features, title):
    sns.pairplot(data[features], hue="label", diag_kind="hist")
    plt.savefig(title, dpi=300, bbox_inches="tight")
    plt.close()


# PRINT feature importance


def print_feature_importance(data, feature_cols):
    X = data[feature_cols]
    y = data["label"].map({"Left": 0, "Right": 1})

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
    model.fit(X, y)

    importances = pd.Series(
        model.feature_importances_,
        index=feature_cols
    ).sort_values(ascending=False)

    #### bar plot
    plt.figure(figsize=(7, 5))
    importances.plot(kind="barh")
    plt.title("Feature Importance (Left vs Right)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("Feature_Importances", dpi=300)
    plt.close()


# Main

def main():
    l_df, r_df, data = load_datasetes()

    # Clean column names again
    data.columns = data.columns.str.strip()

    # Safely drop time column
    data = data.drop(columns=["time (s)"], errors="ignore")

    # ---- PRINT important features ----
    feature_cols = [
        "acc_x", "acc_y", "acc_z",
        "gyro_x", "gyro_y", "gyro_z"
    ]
    print_feature_importance(data, feature_cols)


    plot_timeseries(l_df, r_df, ["acc_x", "acc_y", "acc_z"], "Acceleration Magnitude")
    plot_timeseries(l_df, r_df, ["gyro_x", "gyro_y", "gyro_z"], "Gyroscope Magnitude")


    plot_single_axis_timeseries(
        l_df, r_df, ["acc_x", "acc_y", "acc_z"], "Acceleration (m/s^2)"
    )
    plot_single_axis_timeseries(
        l_df, r_df, ["gyro_x", "gyro_y", "gyro_z"], "Gyroscope (rad/s)"
    )

    plot_pairplot(data, ["acc_x", "acc_y", "acc_z", "label"], "acc_pairplot")



if __name__ == "__main__":
    main()
