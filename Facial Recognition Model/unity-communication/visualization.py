'''
import matplotlib.pyplot as plt
from threading import Thread

# Dictionary to store emotion-specific probabilities
emotion_probabilities = {
    'sad': [],
    'mad': [],
    'happy': [],
    'anxious': []
}

# Function to update the probability for a specific emotion
def update_emotion_probability(emotion, probability):
    emotion = emotion.lower()  # Ensure lowercase for consistency
    if emotion in emotion_probabilities:
        emotion_probabilities[emotion].append(probability)
    else:
        print(f"Emotion '{emotion}' not recognized.")

# Function to plot the line graph for each emotion
def plot_emotion_graphs():
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Emotion Probabilities Over Time')

    # Create line objects for each emotion
    lines = {}
    for emotion, _ in emotion_probabilities.items():
        lines[emotion] = axs[divmod(list(emotion_probabilities.keys()).index(emotion), 2)].plot([], [], label=emotion)[0]

    # Set common labels and titles
    for ax_row in axs:
        for ax in ax_row:
            ax.set_xlabel('Time')
            ax.set_ylabel('Probability')

    while True:
        # Update each line with new data
        for emotion, line in lines.items():
            line.set_data(range(len(emotion_probabilities[emotion])), emotion_probabilities[emotion])

        # Adjust plot limits and pause for 1 second before updating the plot
        for ax_row in axs:
            for ax in ax_row:
                ax.relim()
                ax.autoscale_view(True,True,True)
        
        plt.tight_layout()
        plt.pause(1)  # Pause for 1 second before updating the plot


def start_plotting_thread():
    plot_thread = Thread(target=plot_emotion_graphs)
    plot_thread.daemon = True  # Daemonize the thread so it automatically closes when the main thread exits
    plot_thread.start()

# You should call start_plotting_thread() before running the server
start_plotting_thread()
'''