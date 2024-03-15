import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_table_data
import streamlit as st
from langchain_community.callbacks import get_openai_callback
from src.mcqgenerator.main import generate_evaluate_chain
from src.mcqgenerator.logger import logging

#loading the json file
with open('/Users/sohailshaik/mcqgen/response.json', 'r') as file:
    response_json = json.load(file)

#creating a title for the app
st.title("MCQs Creator Application with LangChain 🧠💡")

#Create a form using st.form
with st.form("user_inputs"):
    #File Upload
    uploaded_file=st.file_uploader("Upload a PDF or txt file")

    #Input Fields
    mcq_count=st.number_input("No. of MCQs", min_value=3, max_value=50)

    #Subject
    subject=st.text_input("Insert Subject", max_chars=20)

    # Quiz Tone
    tone=st.text_input("Complexity Level Of Questions", max_chars=20, placeholder="Simple")

    #Add Button
    button=st.form_submit_button("Create MCQs")

# Check if the button is clicked and all fields have input
if button and uploaded_file is not None and mcq_count and subject and tone:
    with st.spinner("Loading... "):
        try:
            text=read_file(uploaded_file)
            #Count tokens and the cost of API call
            with get_openai_callback() as cb:
                response=generate_evaluate_chain(
                    {
                        "n_text": text,
                        "n_number": mcq_count,
                        "n_subject": subject,
                        "n_tone": tone,
                        "n_response_json": json.dumps(response_json)
                    }
                )
            #st.write(response)
        except Exception as e:
            traceback.print_exception(type(e), e, e.__traceback__)
            st.error("Error")

        else:
            print(f"Total Tokens:{cb.total_tokens}")
            print(f"Prompt Tokens:{cb.prompt_tokens}")
            print(f"Completion Tokens:{cb.completion_tokens}")
            print(f"Total Cost:{cb.total_cost}")
            if isinstance(response, dict):
                # Extract the quiz data from the response
                quiz=response.get("n_quiz", None)
                if quiz is not None:
                    quiz=json.loads(quiz)
                    quiz_table_data = []
                    for key, value in quiz.items():
                        mcq = value["mcq"]
                        options = " | ".join(
                            [
                                f"{option}: {option_value}"
                                for option, option_value in value["options"].items()
                                ]
                            )
                        correct = value["correct"]
                        quiz_table_data.append({"MCQ": mcq, "Choices": options, "Correct": correct})
                    df =pd.DataFrame(quiz_table_data)
                    st.table(df)
                    # Display the review in a text box as well
                    st.text_area(label="Review", value=response["n_review"])
                        
                else:
                    st.error("Error in the table data")
            else:
                st.write(response)
            





