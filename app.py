import streamlit as st
from dotenv import load_dotenv  # Import the load_dotenv function from the dotenv library
import os  # Import the os module for interacting with the operating system
import boto3
from call_processing import upload_to_s3, transcribe_file, customer_prompt_llm, meeting_prompt_llm



# Loading environment variables from the .env file
load_dotenv()
# Setting up the default AWS session using the profile name from the environment variable
boto3.setup_default_session(profile_name=os.getenv('profile_name'))

# Setting up the Bedrock client with a custom timeout configuration
bedrock_client = boto3.client('bedrock-runtime', os.getenv('region'))

### Title displayed on the Streamlit Web App
st.set_page_config(page_title="Call Summary", page_icon=":tada", layout="wide")


#Header and Subheader dsiplayed in the Web App
with st.container():
    st.header("Analyize Audio with Generative AI")
    st.subheader("")
    st.title("Customer Support Call Summarization")



# Workflow Dropdown - this determines the file to operate on, and the LLM prompt to use
workflow_options = ['Customer Service Call', 'Meeting Summary']
workflow = st.selectbox("Select a Workflow", workflow_options)




bucket=os.getenv('bucket')
if "bucket" not in st.session_state:
    st.session_state["bucket"] = bucket

go = st.button("Analyze Audio")
if go:
    if workflow =='Customer Service Call':
        audio_file = "test-audio.m4a"
    elif workflow == 'Meeting Summary':
        audio_file = "bedrock-meeting.m4a"
    else:
        audio_file = "default"


    with st.status("Processing PDF", expanded=False, state="running") as status:
        #upload to s3
        status.update(label=f"Uploading Audio File {audio_file} to S3", state="running", expanded=False)
        results = upload_to_s3(audio_file,audio_file)
        st.write(f":heavy_check_mark: {audio_file} uploaded to S3")

        #Transcribe Job
        status.update(label=f"Starting Transcribe Job for {audio_file} - can take around 30 seconds", state="running", expanded=False)
        transcript = transcribe_file(audio_file)
        st.write(f":heavy_check_mark: {audio_file} Transcribed")
        status.write(f"Full transcription: {transcript}", state="running")

        status.update(label=f"Extracting Summary for {audio_file} with {workflow} prompt", state="running", expanded=False)

        #Branch based on selected workflow
        if workflow == 'Customer Service Call':
            llm_results =customer_prompt_llm(bedrock_client, transcript)
        else:
            llm_results =meeting_prompt_llm(bedrock_client, transcript)
        #LLM
        
        status.update(label=f"Summary Generated", state="complete", expanded=False)
        st.markdown(f"Summary: {llm_results}")

    st.markdown(llm_results)

