import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
import warnings

from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# -----------------------------
# 1. Simulate Healthcare Traffic Data
# -----------------------------
def simulate_healthcare_traffic(num_samples=1000, attack_ratio=0.2):
    np.random.seed(42)
    normal_data = np.random.normal(loc=50, scale=10, size=(int(num_samples * (1 - attack_ratio)), 4))
    attack_data = np.random.normal(loc=100, scale=20, size=(int(num_samples * attack_ratio), 4))  # Simulated attacks

    X = np.vstack((normal_data, attack_data))
    y = np.hstack((np.zeros(normal_data.shape[0]), np.ones(attack_data.shape[0])))  # 0: normal, 1: attack

    df = pd.DataFrame(X, columns=["Packets", "Duration", "Size", "Interval"])
    df["Label"] = y
    return df

# -----------------------------
# 2. Hybrid Detection Model
# -----------------------------
class HybridDetector:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.2, random_state=42)
        self.kmeans = KMeans(n_clusters=2, random_state=42)

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.kmeans.fit(X_scaled)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        iso_pred = self.isolation_forest.predict(X_scaled)
        kmeans_pred = self.kmeans.predict(X_scaled)

        iso_pred = np.where(iso_pred == -1, 1, 0)
        kmeans_pred = np.where(kmeans_pred == kmeans_pred[0], 0, 1)  # Cluster label conversion

        final_pred = np.where((iso_pred + kmeans_pred) >= 1, 1, 0)
        return final_pred

# -----------------------------
# 3. Reinforcement Learning Adaptation Engine (Q-Learning)
# -----------------------------
class AdaptiveRLAgent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.1

    def get_state(self, precision, recall):
        return f"{round(precision, 1)}_{round(recall, 1)}"

    def choose_action(self, state):
        if state not in self.q_table:
            self.q_table[state] = {'tune_up': 0, 'tune_down': 0, 'keep': 0}
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(['tune_up', 'tune_down', 'keep'])
        return max(self.q_table[state], key=self.q_table[state].get)

    def update_q(self, state, action, reward, next_state):
        if next_state not in self.q_table:
            self.q_table[next_state] = {'tune_up': 0, 'tune_down': 0, 'keep': 0}
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + \
                                      self.alpha * (reward + self.gamma * max(self.q_table[next_state].values()))

    def adapt_model(self, detector, action):
        if action == 'tune_up':
            detector.isolation_forest.set_params(contamination=min(0.5, detector.isolation_forest.contamination + 0.01))
        elif action == 'tune_down':
            detector.isolation_forest.set_params(contamination=max(0.01, detector.isolation_forest.contamination - 0.01))
        # 'keep' does nothing

# -----------------------------
# 4. Evaluation Function
# -----------------------------
def evaluate_model(y_true, y_pred, latency):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, recall, f1, latency

# -----------------------------
# 5. Real-Time Simulation Loop
# -----------------------------
def real_time_adaptive_simulation(df, episodes=20):
    detector = HybridDetector()
    agent = AdaptiveRLAgent()

    logs = []
    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}/{episodes}")
        sampled_df = df.sample(frac=0.2)
        X, y = sampled_df.drop("Label", axis=1), sampled_df["Label"]

        start_time = time.time()
        detector.fit(X)
        preds = detector.predict(X)
        latency = time.time() - start_time

        precision, recall, f1, latency = evaluate_model(y, preds, latency)
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}, Latency: {latency:.4f}s")

        state = agent.get_state(precision, recall)
        action = agent.choose_action(state)
        reward = f1  # Reward can be modified

        agent.adapt_model(detector, action)
        new_preds = detector.predict(X)
        new_precision, new_recall, new_f1, _ = evaluate_model(y, new_preds, 0)
        next_state = agent.get_state(new_precision, new_recall)
        agent.update_q(state, action, reward, next_state)

        logs.append({
            "episode": episode + 1,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "latency": latency,
            "action": action
        })

    return pd.DataFrame(logs)

# -----------------------------
# 6. Visualize Results
# -----------------------------
def plot_results(results):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    plt.plot(results['episode'], results['precision'], marker='o', label='Precision')
    plt.plot(results['episode'], results['recall'], marker='s', label='Recall')
    plt.plot(results['episode'], results['f1'], marker='^', label='F1 Score')
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.title("Cyber Resilience System Performance Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7. Ethical Safeguards Notice
# -----------------------------
def display_ethics_notice():
    print("\n[Ethical Safeguards]")
    print("✔️ This simulation uses anonymized, synthetic data only.")
    print("✔️ No real patient or PII data was used.")
    print("✔️ Designed for HIPAA and GDPR-compliant research simulations.")

# -----------------------------
# 8. Main Execution
# -----------------------------
if __name__ == "__main__":
    print("Initializing Real-Time Adaptive Cyber Resilience System...")
    display_ethics_notice()
    traffic_df = simulate_healthcare_traffic(num_samples=2000)
    results = real_time_adaptive_simulation(traffic_df, episodes=20)
    plot_results(results)
    print("\nSystem adaptation and performance evaluation complete.")
