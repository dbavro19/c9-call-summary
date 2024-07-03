import boto3
import json
import botocore
import streamlit as st
import time
import requests



#Invoke LLM - Bedrock - Creates the Docuemnt based on user feedback
def invoke_llm_summary(bedrock, transcript):

    print("TESTING TRANSCRIPTION IN INVOKE")
    print(transcript)




    template="""
Full Call Log: 
(Full text of the Call transcription broken up line by line by speaker - Each line should be sperarted by a new line )
[EXAMPLE]
Customer Support Agent: (Call Text)
Customer: (Call Text)
Customer Support Agent: (Call Text)
[etc.]
[/EXAMPLE]

###

Customer Support Agent Name: (Name of the Customer Support Agent, "Not Provided" of name is not provided)

Cusotmer Name: (Name of the Customer, "Not Provided" of name is not provided)

Customer Issue: (Detailed Summary of the Customer Issue)

Actions Taken: (Detailed Summary of the Actions Taken)

Resolution: (Summary of the Resolution)

Sentiment Analysis: (Provide a Sentiment Analysis of the Cusotmers Sentiament based on the result of the call. Should be one of Positive, Mixed, Negative)"""

# Uses the Bedrock Client, the user input, and the document template as part of the prompt


    ##Setup Prompt
    prompt_data = f"""

Human:

Generate a summary of the customer support phone call provided in the <call_transcript> XML tags
Your repsonse should follow the format and structure of the provided template
Base your response soley on the content of the call transcript, dont make too many assumptions
Response should be in valid markdown format, Section Titles should be Bold

###

<Call_Transcript>
{transcript}
</Call_Transcript>

<Output_Template>
{template}
</Output_Template>

###

Assistant: Here is a summary of the call

"""
#Add the prompt to the body to be passed to the Bedrock API
#Also adds the hyperparameters 

    body = json.dumps({"prompt": prompt_data,
                 "max_tokens_to_sample":5000,
                 "temperature":.2,
                 "stop_sequences":[]
                  }) 
    
    #Run Inference
    modelId = "anthropic.claude-v2"  # change this to use a different version from the model provider if you want to switch 
    accept = "application/json"
    contentType = "application/json"
    #Call the Bedrock API
    response = bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    #Parse the Response
    response_body = json.loads(response.get('body').read())
    llmOutput=response_body.get('completion')

    print(llmOutput)

    #Return the LLM response
    return llmOutput



def customer_prompt_llm(bedrock, content):

    system_prompt = f"""
You are a customer service agent, and you are tasked with providing a summary of a customer support call.
You will be given the call transcript, and you will need to extract the following information:
    The Full call transcript broken down line by line by the speaker
    The Customer Support Agent Name
    The Customer Name
    Customer Issue Summary
    Product Name (if applicable)
    Actions Taken
    Resolution
    Sentiment Analysis of the Customer


<example_format>
Full Call Log: 
(Full text of the Call transcription broken up line by line by speaker - Every time the speaker changes, capture that as a new line -  Each line should be separated by a new line )
Customer Support Agent: (Call Text)

Customer: (Call Text)

Customer Support Agent: (Call Text)

[etc.]
[/EXAMPLE]

###
DETAILS
###

Customer Support Agent Name: (Name of the Customer Support Agent, "Not Provided" of name is not provided)

Customer Name: (Name of the Customer, "Not Provided" of name is not provided)

Customer Issue: (Detailed Summary of the Customer Issue)

Product Name: (If applicable, the name of the product the customer is interacting with)

Actions Taken: (Detailed Summary of the Actions Taken)

Resolution: (Summary of the Resolution)

Sentiment Analysis: (Provide a Sentiment Analysis of the Cusotmers Sentiament based on the result of the call. Should be one of Positive, Mixed, Negative)
</example_format>

Your response should be in valid markdown (with section titles in Bold)

return the full summary with the extracted details in <output> xml tags, only including the requested markdown in your response

"""

    
    # Create the prompt dictionary
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<customer_call_transcript> {content} </customer_call_transcript>"
            }
        ]
    }

    # Convert the prompt to a JSON string
    prompt = json.dumps(prompt)


    # Invoke the Bedrock model with the prompt
    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput = response_body['content'][0]['text']

    # Parse the scratchpad and output from the LLM response
    output = parse_xml(llmOutput, "output")

    return output




def meeting_prompt_llm(bedrock, content):
    print("TESTING TRANSCRIPTION IN INVOKE")

    system_prompt = f"""
You are an Executive Assistant Agent. You are tasked with providing a summary of a meeting.
You will be given the meeting transcript, and you will need to extract the following information - Dont make assumptions, if you cant find/identify any of the requested details reply "Not Provided" for the specific detail:
    Detailed Meeting Summary
    The Attendees
    The Agenda
    The Actions Items
    The Sentiment Analysis of the Attendees






<example_format>
Meeting Summary: (Provide a detailed summary of the meeting)

Attendees: (List all attendees, and their roles)

Agenda: (Provide a concise breakdown of the meeting agenda)

Actions Items: (List all action items, who owns them, their current status, and their due dates)

Meeting Sentiment: (Provide a Sentiment Analysis of the Attendees based on the result of the meeting. Should be one of Positive, Mixed, Negative)


</example_format>

Return your response in valid markdown with section titles in Bold

return the full summary with the extracted details in <output> xml tags, only including the requested markdown in your response

"""

    
    # Create the prompt dictionary
    prompt = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10000,
        "temperature": 0.5,
        "system": system_prompt,
        "messages": [    
            {
                "role": "user",
                "content": f"<customer_call_transcript> {content} </customer_call_transcript>"
            }
        ]
    }

    # Convert the prompt to a JSON string
    prompt = json.dumps(prompt)

    # Invoke the Bedrock model with the prompt
    response = bedrock.invoke_model(body=prompt, modelId="anthropic.claude-3-sonnet-20240229-v1:0", accept="application/json", contentType="application/json")
    response_body = json.loads(response.get('body').read())
    llmOutput = response_body['content'][0]['text']



    # Parse the scratchpad and output from the LLM response
    output = parse_xml(llmOutput, "output")

    return output


def generate_summary(transcript):

    #Setup bedrock client
    bedrock = boto3.client('bedrock-runtime' , 'us-west-2')

    #invoke LLM
    llmOutput = invoke_llm_summary(bedrock, transcript)
    return llmOutput



def transcribe_file(object_name):

    transcribe_client = boto3.client('transcribe')
    bucket_name = st.session_state.bucket

    file_uri= f"s3://{bucket_name}/{object_name}"
    job_name=object_name+time.strftime("%Y%m%d-%H%M%S")
    full_transcript=""

    transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={'MediaFileUri': file_uri},
        #MediaFormat='wav',
        LanguageCode='en-US'
    )

    max_tries = 60
    while max_tries > 0:
        max_tries -= 1
        job = transcribe_client.get_transcription_job(TranscriptionJobName=job_name)
        job_status = job['TranscriptionJob']['TranscriptionJobStatus']
        if job_status in ['COMPLETED', 'FAILED']:
            print(f"Job {job_name} is {job_status}.")
            if job_status == 'COMPLETED':
                print(
                    f"Download the transcript from\n"
                    f"\t{job['TranscriptionJob']['Transcript']['TranscriptFileUri']}.")
                
                job_result = requests.get(
                    job['TranscriptionJob']['Transcript']['TranscriptFileUri']).json()
                full_transcript=job_result['results']['transcripts'][0]['transcript']


                return full_transcript
            break
        else:
            print(f"Waiting for {job_name}. Current status is {job_status}.")
        time.sleep(10)



def upload_to_s3(file_name, object_name):
    bucket=st.session_state.bucket


    s3 =boto3.client('s3')
    response = s3.upload_file(file_name, bucket, object_name)

    return object_name

def parse_xml(xml, tag):
    """
    Helper function to extract the value between XML tags from a given string.
    """
    start_tag = f"<{tag}>"
    end_tag = f"</{tag}>"
    
    start_index = xml.find(start_tag)
    if start_index == -1:
        return ""

    end_index = xml.find(end_tag)
    if end_index == -1:
        return ""

    value = xml[start_index+len(start_tag):end_index]
    return value

