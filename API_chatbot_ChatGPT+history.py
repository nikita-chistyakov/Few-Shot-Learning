import streamlit as st
import requests

# OpenAI API
API_URL = "https://api.openai.com/v1/chat/completions"
headers = {"Authorization": "Bearer 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"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def main():
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

if __name__ == "__main__":
    main()



