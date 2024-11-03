import streamlit as st
from llm1 import create_chatbot, task


st.title(f"{task.title()} Culture Chatbot")
st.markdown("*Share your experiences and cultural traditions!*")

# Initialize chatbot and chat history in session state
if "chatbot" not in st.session_state:
    st.session_state.chatbot = create_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Generate first question
    initial_question = st.session_state.chatbot.generate_next_question()
    st.session_state.messages.append({
        "role": "assistant",
        "content": initial_question,
        "current_question": initial_question
    })

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type your response here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Get the current question that was asked
    current_question = next(
        (msg["current_question"] for msg in reversed(st.session_state.messages) 
         if "current_question" in msg),
        "What festival would you like to tell me about?"
    )
    
    # Update chatbot context with the new Q&A
    st.session_state.chatbot.update_context(current_question, prompt)
    
    # Generate next question
    next_question = st.session_state.chatbot.generate_next_question()
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(next_question)
    
    # Add assistant response to chat history
    st.session_state.messages.append({
        "role": "assistant",
        "content": next_question,
        "current_question": next_question
    })