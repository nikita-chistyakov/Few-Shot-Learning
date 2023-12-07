from keras.src.applications.vgg16 import preprocess_input
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests

# Function to preprocess the query image for Siamese model prediction
def preprocess_query_image(image_path):
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((28, 28))
    gray_image = resized_image.convert('L')
    normalized_image = np.array(gray_image) / 255.0
    return normalized_image

# Function to load and compile the Siamese model
def load_and_compile_siamese_model():
    siamese_model_path = 'C:\\Users\\siamese_model.h5'
    siamese_model = tf.keras.models.load_model(siamese_model_path)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy')
    return siamese_model



# Load the Siamese model and compile it before running the Streamlit app
siamese_model = load_and_compile_siamese_model()

# Define input image size
input_size_crop_classification = (224, 224)

# Load the fine-tuned model from the saved file
model_path = 'C:\\Users\\crop_classification_model_fine_tuned.h5'
crop_classification_model = tf.keras.models.load_model(model_path)

# Define crop labels for displaying results
crop_labels = {
    0: "Cucumber",
    1: "Tobacco-plant",
    2: "jute",
    3: "maize",
    4: "mustard-oil",
    5: "rice",
    6: "wheat"
}

# Define the recommended and alternative soil types for each crop
recommended_soil = {
    'Cucumber': 'Sandy loam, loam',
    'Tobacco-plant': 'Silt loam, clay loam',
    'jute': 'Sandy loam, clay loam',
    'maize': 'Loam, clay loam',
    'mustard-oil': 'Silt loam, loam',
    'rice': 'Clay loam, clay',
    'wheat': 'Silt loam, clay loam'
}

alternative_soil = {
    'Cucumber': 'Clay, silt, loam',
    'Tobacco-plant': 'Clay, loam',
    'jute': 'Sandy soil, loam',
    'maize': 'Sandy loam, loam',
    'mustard-oil': 'Clay loam, loam',
    'rice': 'Loam, sandy loam',
    'wheat': 'Sandy loam, clay loam'
}

#Function to preprocess the query image for prediction
# Function to preprocess the query image for Siamese model prediction
def preprocess_query_image(image_path):
    image = Image.open(image_path).convert('RGB')
    resized_image = image.resize((28, 28))
    gray_image = resized_image.convert('L')
    normalized_image = np.array(gray_image) / 255.0
    return normalized_image

# Function to predict the crop class from the query image
def predict_crop_class(image_path):
    query_image = Image.open(image_path).resize(input_size_crop_classification)
    query_image_array = tf.keras.preprocessing.image.img_to_array(query_image)
    query_image_array = tf.expand_dims(query_image_array, 0)
    processed_query_img = tf.keras.applications.vgg16.preprocess_input(query_image_array)
    predictions = crop_classification_model.predict(processed_query_img)
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_crop = crop_labels[predicted_class]
    return predicted_crop

# Function to get recommended and alternative soil types for the predicted crop
def get_soil_types(predicted_crop):
    recommended_soil_type = recommended_soil.get(predicted_crop, 'Not available')
    alternative_soil_type = alternative_soil.get(predicted_crop, 'Not available')
    return recommended_soil_type, alternative_soil_type

# Function to load and compile the Siamese model
def load_and_compile_siamese_model():
    siamese_model_path = 'C:\\Users\\siamese_model.h5'
    siamese_model = tf.keras.models.load_model(siamese_model_path)
    siamese_model.compile(optimizer='adam', loss='binary_crossentropy')
    return siamese_model

# Load the Siamese model and compile it before running the Streamlit app
siamese_model = load_and_compile_siamese_model()

def query(payload):
    # OpenAI API
    API_URL = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": "Bearer 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"}

    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def chatbot_page():
    st.title("Chatbot at your service 💬")
    st.write("Ask your questions below:")

    # Initialize or get the chat history from session state
    chat_history = st.session_state.get("chat_history", [])

    user_input = st.text_input("You:")
    if user_input:
        if user_input.lower() == "exit":
            st.write("Chatbot: Goodbye!")
        else:
            # Prepare payload for ChatGPT API with the desired model and temperature
            payload = {
                "model": "gpt-3.5-turbo",  # Replace with the desired model name
                "messages": [{"role": "user", "content": user_input}],
                "max_tokens": 1500,
                "temperature": 0.8,
            }

            # Make the API request
            response = query(payload)
            print(response)  # Print the API response (for debugging)

            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["message"]["content"]
                st.write("Chatbot:", generated_text)

                # Store the user input and chatbot response in chat history
                chat_history.append({"User": user_input, "Chatbot": generated_text})

                # Keep only the last 5 chat history entries
                chat_history = chat_history[-5:]  # Keep last 5 entries

            else:
                st.write("Chatbot: Sorry, I couldn't generate a response.")

    # Store the updated chat history in session state
    st.session_state.chat_history = chat_history

    # Display the last 5 chat history entries
    st.subheader("Recent messages")
    for chat in reversed(chat_history):
        st.text(f"User: {chat['User']}")
        st.text(f"Chatbot: {chat['Chatbot']}")
        st.text("------------------")

def main():
    # Set the title and sidebar options
    st.title("Crop Classification and Similar Image Prediction")
    page = st.sidebar.selectbox("Select a page", ["Pre-trained Model", "Siamese Model Prediction", "Chatbot"])
    # page one ////////////////////////////////////////////////
    if page == "Pre-trained Model":
        st.header("Crop Classification")
    # Add image upload button
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key='page1')

    # Add search button
        search_button = st.button("Search")

    # Perform crop classification when search button is clicked and an image is uploaded
        if search_button and uploaded_image:
        # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make predictions using the model
            predicted_crop = predict_crop_class(uploaded_image)

        # Get recommended and alternative soil types
            recommended_soil_type = recommended_soil.get(predicted_crop, 'Not available')
            alternative_soil_type = alternative_soil.get(predicted_crop, 'Not available')

        # Display the result
            st.header("Crop Details")
            st.subheader("Predicted Crop:")
            st.write(predicted_crop)
            st.subheader("Recommended Soil Type:")
            st.write(recommended_soil_type)
            st.subheader("Alternative Soil Type:")
            st.write(alternative_soil_type)


    # Second Page: Siamese Model Prediction
    elif page == "Siamese Model Prediction":
        st.header("Siamese Model Prediction")
        st.write("Upload a query image and a set of 3 support images to find the most similar image.")

    # Add query image upload button
        uploaded_query_image = st.file_uploader("Upload the query image", type=["jpg", "jpeg", "png"])

    # Add support image upload buttons
        uploaded_support_images = []
        for i in range(3):
            uploaded_support_image = st.file_uploader(f"Upload support image {i+1}", type=["jpg", "jpeg", "png"])
            if uploaded_support_image is not None:
                uploaded_support_images.append(uploaded_support_image)

    # Add a button to trigger the prediction
        predict_button = st.button("Find Similar Image")

        if predict_button and uploaded_query_image and len(uploaded_support_images) == 3:
        # Display the query image
            st.image(uploaded_query_image, caption="Query Image", use_column_width=True)

        # Preprocess the query image for Siamese model prediction
            preprocessed_query_image = preprocess_query_image(uploaded_query_image)

        # Preprocess the support images for Siamese model prediction
            preprocessed_support_images = []
            for support_image in uploaded_support_images:
                preprocessed_support_image = preprocess_query_image(support_image)
                preprocessed_support_images.append(preprocessed_support_image)

        # Calculate the similarity score for each image in the support set
            similarity_scores = []
            for support_image in preprocessed_support_images:
                siamese_predictions = siamese_model.predict([np.expand_dims(preprocessed_query_image, axis=0), np.expand_dims(support_image, axis=0)])
                similarity_score = siamese_predictions[0][0]
                similarity_scores.append(similarity_score)

        # Find the index of the most similar image in the support set
            most_similar_index = np.argmax(similarity_scores)

        # Retrieve the most similar image from the support set
            most_similar_image = uploaded_support_images[most_similar_index]

        # Display the most similar image and its corresponding crop name
            st.header("Most Similar Image from Support Set")
            st.image(most_similar_image, caption="Most Similar Image", use_column_width=True)

        # Predict the crop class for the query image
            predicted_crop = predict_crop_class(uploaded_query_image)

        # Display the predicted crop name
            st.header("Predicted Crop:")
            st.write(predicted_crop)
    # Third Page: Chatbot
    elif page == "Chatbot":
        chatbot_page()

if __name__ == "__main__":
    main()
