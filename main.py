import streamlit as st
import model

# Run the Streamlit app
def main():
    st.title("VideoAskRAG: Answer your questions from the video 🎥😀")
    st.write("This app allows you to ask questions about a YouTube video and get answers from the video itself.")

    video_url = st.text_input(
        label="Enter the link of your YouTube video here", 
        placeholder="URL",
        max_chars=60,
    )

    # If the URL is valid, display the video and parse the video and ask the user to input a question
    if video_url and "https://www.youtube.com/watch?v=" in video_url:
        st.video(video_url)

        # Ask user to input question
        user_input = st.text_input(
            label="Ask your question here",
            placeholder="Input",
        )


if __name__ == "__main__":
    main()