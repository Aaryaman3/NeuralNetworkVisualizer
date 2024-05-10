import requests
import json
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

URL = 'http://127.0.0.1:5000/'  # Change to the appropriate URL

st.title('Neural Network Visualizer')

st.sidebar.markdown('## Your Image Here ##')

if st.button('See what the neural network thinks!'):
    response = requests.post(URL, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))
    st.sidebar.image(image, width=150)

    # Introduction
    st.write("A neural network is a machine learning model inspired by the human brain. It consists of layers of interconnected nodes called neurons, which process and transmit information. Let's visualize how this neural network analyzes the input image!")

    # Visualize layers with simpler language and visual cues
    for layer_number, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        fig, ax = plt.subplots(figsize=(10, 4))

        # Simpler layer titles and descriptions
        if layer_number == 0:
            layer_title = "Input Layer"
            st.write("The input layer represents the raw image data.")
        elif layer_number == 1:
            layer_title = "Level 1: Looking for patterns"
            st.write("This convolutional layer looks for simple patterns like edges and curves in the input image.")
        elif layer_number == 2:
            continue
        elif layer_number == 3:
            layer_title = "Level 2: Combining patterns"
            st.write("This layer combines the patterns detected in the previous layer to form more complex features.")
        elif layer_number == 4:
            layer_title = "Level 3: Making a guess"
            st.write("The fully connected layer takes the combined features and makes a prediction about the input image.")
        else:
            layer_title = "Final Answer"
            st.write("This is the final output layer, indicating the neural network's prediction for the input image.")

        row = 1  # Adjust rows based on number of neurons
        col = len(numbers)

        # Colored circles for neurons with labels
        cmap = plt.cm.get_cmap('coolwarm')
        for i, number in enumerate(numbers):
            ax = plt.subplot(row, col, i + 1)
            circle = plt.Circle((0.5, 0.5), 0.3, color=cmap(number))
            ax.add_patch(circle)
            ax.text(0.5, 0.7, f"Neuron {i+1}", ha='center', fontsize=8)
            plt.xticks([])
            plt.yticks([])

        # Simpler explanation
        plt.suptitle(layer_title, fontsize=16)
        st.pyplot(fig)

    st.write("That's how the neural network processed the input image step by step! Each layer performed a specific task, building up from simple patterns to complex features, and finally making a prediction.")