import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import random
import gym
from gym import spaces
import pickle  # For saving/loading the RL model
import sys
from sklearn.metrics import confusion_matrix, roc_curve, auc
from collections import deque  # For implementing a rolling window

sys.stdout.flush()
# Load datasets
healthcare_df = pd.read_csv("healthcare_dataset.csv")
drugs_product_df = pd.read_csv("drugs_product.csv")
diabetes_df = pd.read_csv("diabetes.csv")

# Dictionary to store datasets
datasets = {
    "Diabetes Dataset": diabetes_df,
    "Healthcare Dataset": healthcare_df,
    "Drugs Product Dataset": drugs_product_df,
}


# Define RL Environment
class AnomalyDetectionEnv(gym.Env):
    def __init__(self, data):
        super(AnomalyDetectionEnv, self).__init__()
        self.data = data
        self.original_data = data.copy()  # Store original data for resetting
        self.current_index = 0
        self.contamination = 0.05  # Initial contamination threshold
        self.action_space = spaces.Discrete(3)  # {0: Decrease, 1: Increase, 2: No Change}
        self.observation_space = spaces.Box(
            low=np.min(data, axis=0).astype(np.float32),
            high=np.max(data, axis=0).astype(np.float32),
            dtype=np.float32,
        )
        self.threat_level = 0  # Initialize threat level
        self.episode_steps = 0  # Keep track of steps in current episode
        self.time_window = deque(
            maxlen=10
        )  # Rolling window for recent data points (last 10 steps)
        self.recent_anomalies = (
            0  # Count of anomalies in the recent time window, used for reward
        )

    def reset(self, **kwargs):
        self.data = self.original_data.copy()  # Reset to the original data
        self.current_index = 0
        self.contamination = 0.05
        self.threat_level = 0  # Reset threat level
        self.episode_steps = 0
        self.time_window.clear()
        self.recent_anomalies = 0
        return self.data[self.current_index]

    def step(self, action):
        # Adjust contamination based on action
        if action == 0:
            self.contamination = max(0.01, self.contamination - 0.01)  # Decrease
        elif action == 1:
            self.contamination = min(0.2, self.contamination + 0.01)  # Increase
        # Action 2: No change in contamination

        # Simulate threat injection (more dynamic and realistic)
        self.threat_level = max(
            0, self.threat_level + random.uniform(-0.05, 0.15)
        )  # Slower fluctuation
        if random.random() < self.threat_level:
            anomaly_index = random.randint(0, len(self.data) - 1)
            # More subtle, but potentially more impactful anomalies (realistic)
            self.data[anomaly_index] = self.data[anomaly_index] * (
                1 + random.uniform(0.5, 1.2)
            )  # Smaller change

        # Apply Isolation Forest with new contamination threshold
        iso_forest = IsolationForest(
            contamination=self.contamination, random_state=42
        )
        predictions = iso_forest.fit_predict(self.data)
        current_prediction = predictions[self.current_index]  # Get prediction for current step

        # Update rolling window and anomaly count
        self.time_window.append(self.data[self.current_index])
        if current_prediction == -1:
            self.recent_anomalies += 1
        else:
            self.recent_anomalies = max(
                0, self.recent_anomalies - 1
            )  # Decay over time

        # --- Improved Reward Function ---
        # 1. Anomaly Percentage Reward: Target 5% anomaly rate
        anomaly_percentage = (predictions == -1).mean()
        reward_anomaly_percentage = -abs(anomaly_percentage - 0.05)

        # 2. Threat Level Penalty: Higher threat level, lower reward
        reward_threat = -self.threat_level * 0.5  # Scale it down

        # 3. Recent Anomaly Penalty: More recent anomalies, lower reward (more immediate feedback)
        reward_recent_anomalies = -self.recent_anomalies * 0.2  # Scale it

        # 4. Action Cost: Small penalty for changing contamination too frequently
        reward_action = 0
        if action != 2:  #if action is not "no change"
            reward_action = -0.01

        # Combine rewards (weighted sum)
        reward = (
            reward_anomaly_percentage + reward_threat + reward_recent_anomalies + reward_action
        )

        # --- End Improved Reward ---

        # Move to next record, avoid infinite loops
        self.current_index = (self.current_index + 1) % len(self.data)
        self.episode_steps += 1
        done = (
            self.current_index == 0 or self.episode_steps >= 100
        )  # Episode ends after one pass or 100 steps

        return self.data[self.current_index], reward, done, {
            "threat_level": self.threat_level
        }


# Optimized RL Training Function (Pre-Trained Model)
def train_rl_agent(env, episodes=50):
    q_table_path = "q_table.pkl"  # File to save/load Q-table

    # If model exists, load it instead of training
    try:
        with open(q_table_path, "rb") as file:
            q_table = pickle.load(file)
        print(" Loaded pre-trained RL model.")
        return q_table
    except FileNotFoundError:
        print(" Training RL model from scratch...")

    q_table = np.zeros((len(env.data), env.action_space.n))
    learning_rate = 0.1
    discount_factor = 0.9
    epsilon = 1.0  # Initial exploration rate
    epsilon_decay_rate = 0.99  # Decay epsilon over episodes
    min_epsilon = 0.01

    for episode in range(episodes):
        state = env.reset()
        done = False
        step_count = 0
        max_steps = 100  # Increased steps per episode

        print(f"Training episode {episode + 1}/{episodes}")  # Debugging progress

        while not done and step_count < max_steps:
            step_count += 1

            # Choose action
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(
                    q_table[env.current_index]
                )  # Exploit - use current index

            # Take action
            next_state, reward, done, info = env.step(action)

            # Update Q-table
            q_table[env.current_index, action] = (1 - learning_rate) * q_table[
                env.current_index, action
            ] + learning_rate * (
                reward + discount_factor * np.max(q_table[env.current_index])
            )  # use current index

        epsilon = max(
            min_epsilon, epsilon * epsilon_decay_rate
        )  # Faster exploration decay, and minimum epsilon

    # Save the trained Q-table for future use
    with open(q_table_path, "wb") as file:
        pickle.dump(q_table, file)

    print("RL model training completed and saved.")
    return q_table


# Process each dataset with anomaly detection + RL
results = []


def calculate_metrics(y_true, y_pred, y_scores):
    """
    Calculates and returns a dictionary of evaluation metrics.

    Args:
        y_true (array-like): Ground truth labels (0 for normal, 1 for anomaly).
        y_pred (array-like): Predicted labels (0 for normal, 1 for anomaly).
        y_scores (array-like): Anomaly scores from the Isolation Forest.

    Returns:
        dict: A dictionary containing the calculated metrics.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Handle the case where there are no true positives or true negatives.
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    tnr = tn / (tn + fp) if (fp + tn) > 0 else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr  # Recall is the same as TPR
    f1_score = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    # Print anomaly score information
    print(f"Min anomaly score: {np.min(y_scores)}")
    print(f"Max anomaly score: {np.max(y_scores)}")
    print(f"Mean anomaly score: {np.mean(y_scores)}")
    print(f"Some anomaly scores: {y_scores[:10]}")  # Print the first 10 scores

    # Check if lower scores indicate anomalies and reverse if necessary
    if np.mean(y_scores[y_pred == 1]) < np.mean(y_scores[y_pred == 0]):
        print("Reversing anomaly scores for ROC calculation")
        y_scores_for_roc = -y_scores
    else:
        y_scores_for_roc = y_scores

    # Calculate ROC curve and AUC
    try:
        fpr_roc, tpr_roc, _ = roc_curve(y_true, y_scores_for_roc)
        roc_auc = auc(fpr_roc, tpr_roc)
    except ValueError as e:
        print(f"Error calculating ROC AUC: {e}")
        # Handle the error: set a default value or re-raise
        roc_auc = 0.5  # Default AUC for an uninformative classifier
        fpr_roc, tpr_roc = [0, 1], [0, 1]  # default ROC curve

    return {
        "TPR": tpr,
        "FPR": fpr,
        "FNR": fnr,
        "TNR": tnr,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1_score,
        "ROC AUC": roc_auc,
        "ROC Curve": (fpr_roc, tpr_roc),  # Return the ROC curve data
    }


for dataset_name, df in datasets.items():
    print(f"\n Processing {dataset_name}...")

    # Select only numeric columns
    df_numeric = df.select_dtypes(include=[np.number])

    # Handle missing values (fill with column mean)
    df_numeric.fillna(df_numeric.mean(), inplace=True)

    # Reduce number of columns to avoid RL performance issues
    if df_numeric.shape[1] > 10:
        df_numeric = df_numeric.iloc[:, :10]  # Select only first 10 columns

    # Standardize data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    # Apply K-Means Clustering (2 clusters: normal & suspicious)
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    df_numeric["KMeans_Cluster"] = kmeans.fit_predict(scaled_data)

    # Train RL model to optimize anomaly detection
    env = AnomalyDetectionEnv(scaled_data)
    q_table = train_rl_agent(env)

    # Apply Isolation Forest using optimized contamination threshold
    # Use the average Q-value across all states to decide the best action
    avg_q_values = np.mean(q_table, axis=0)
    best_action = np.argmax(avg_q_values)

    optimal_contamination = env.contamination
    # Simulate the effect of the best action on the initial contamination
    if best_action == 0:
        optimal_contamination = max(0.01, optimal_contamination - 0.01)
    elif best_action == 1:
        optimal_contamination = min(0.2, optimal_contamination + 0.01)
    #if best action is 2, do not change the contamination level

    iso_forest = IsolationForest(
        contamination=optimal_contamination, random_state=42
    )
    predictions = iso_forest.fit_predict(scaled_data)
    df_numeric["Anomaly_Score"] = predictions

    # Calculate Evaluation Metrics
    # Create binary labels: 0 for normal, 1 for anomaly
    y_true = np.zeros(len(df_numeric))
    y_true[df_numeric["Anomaly_Score"] == -1] = 1
    y_pred = (df_numeric["Anomaly_Score"] == -1).astype(int)
    y_scores = iso_forest.decision_function(scaled_data)  # Get anomaly scores

    metrics = calculate_metrics(y_true, y_pred, y_scores)

    # Count percentage of anomalies detected
    anomaly_percentage = (df_numeric["Anomaly_Score"] == -1).mean() * 100
    results.append(
        {
            "Dataset": dataset_name,
            "Total Records": len(df_numeric),
            "Anomalies Detected (%)": round(anomaly_percentage, 2),
            "Optimal Contamination": round(optimal_contamination, 2),
            **metrics,  # Include the calculated metrics in the results
        }
    )

    # Visualization - K-Means Clustering & Anomaly Detection (only for datasets with at least 2 features)
    if scaled_data.shape[1] >= 2:
        plt.figure(figsize=(18, 6))  # Increased figure size for 3 plots

        # K-Means Clustering Plot
        plt.subplot(1, 3, 1)
        sns.scatterplot(
            x=scaled_data[:, 0],
            y=scaled_data[:, 1],
            hue=df_numeric["KMeans_Cluster"],
            palette="viridis",
        )
        plt.title(f"K-Means Clustering - {dataset_name}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        # Isolation Forest Anomaly Detection Plot
        plt.subplot(1, 3, 2)
        sns.scatterplot(
            x=scaled_data[:, 0],
            y=scaled_data[:, 1],
            hue=df_numeric["Anomaly_Score"],
            palette="coolwarm",
        )
        plt.title(f"RL-Optimized Isolation Forest - {dataset_name}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        # ROC Curve Plot
        plt.subplot(1, 3, 3)
        fpr_roc, tpr_roc = metrics["ROC Curve"]  # Get ROC data from metrics
        plt.plot(
            fpr_roc,
            tpr_roc,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {metrics['ROC AUC']:0.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {dataset_name}")
        plt.legend(loc="lower right")

        plt.tight_layout()
        plt.show()
    else:
        print(
            f"Skipping visualization for {dataset_name} as it has less than 2 features."
        )

# Display anomaly detection results
results_df = pd.DataFrame(results)
print("\n RL-Optimized Anomaly Detection Summary:")
print(results_df.to_string(float_format="{:.4f}".format))
