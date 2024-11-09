import streamlit as st
from model import create_transcription, get_response

# Run the Streamlit app
def main():
    st.title("VideoAskRAG: Answer Your Questions From Videos")
    st.write("This app allows you to ask questions about a YouTube video and get answers from the video itself.")

    with st.form(key="video_url_form", border=False):
        video_url = st.text_input(
            label="Enter the link of your YouTube video here", 
            placeholder="URL",
            max_chars=60
        )
        submit_video_url = st.form_submit_button(label="Submit")

    # If the URL is valid, display the video and parse the video and ask the user to input a question,
    # then display the model's response
    if submit_video_url and video_url:
        if "https://www.youtube.com/watch?v=" in video_url:
            # Generate a transcription of the video
            st.video(data=video_url)
            with st.spinner("Transcribing video, please wait before asking a question..."):
                create_transcription(video_url)

            with st.form(key="question_form", border=False):
                user_input = st.text_input(
                    label="Ask your question here", 
                    placeholder="Input"
                )
                submit_question = st.form_submit_button(label="Submit")

            # Once the user inputs a question, display the model's response
            if submit_question and user_input:
                with st.spinner("Please wait while we process your question..."):
                    model_response = get_response(
                        video_url=video_url, 
                        input_text=user_input
                    )
                st.write(f"Answer: {model_response}")

        else:
            st.write("Please enter a valid YouTube video URL.")

if __name__ == "__main__":
    main()