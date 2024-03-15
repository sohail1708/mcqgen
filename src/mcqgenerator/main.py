import os
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain_community.llms import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv


#loading the environment variables from .env
load_dotenv()

#openai key
key = os.getenv("OPENAI_API_KEY")

#creating llm model name and the temperature of the response
llm = OpenAI(openai_api_key = key)


#creating a prompt template to fullfill the task 
TEMPLATE="""
Text:{n_text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {n_number} multiple choice questions for {n_subject} students in {n_tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {n_number} MCQs
### RESPONSE_JSON
{n_response_json}

"""

# the prompt template gets the input values and output the response in json format
quiz_generation_prompt = PromptTemplate(
    input_variables= ["n_text", "n_number", "n_subject", "n_tone", "n_response_json"],
    template= TEMPLATE
)

# Creating a quiz chain using LLMChain. The LLMchain takes two args 1. llm object and 2. the prompt template created

quiz_chain = LLMChain(llm=llm, prompt = quiz_generation_prompt, output_key = 'n_quiz', verbose = True)

# this templates reviews the complexity, grammar and subject relevance of the quiz that is generated from the frist prompt
template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {n_subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{n_quiz}

Check from an expert English Writer of the above quiz:
"""

# this is a prompt template to evaluate the quiz
quiz_evaluation_prompt=PromptTemplate(input_variables=["n_subject", "n_quiz"], template=template2)


# this is a review chain which evaluates the quiz prompts
review_chain = LLMChain(llm=llm, prompt = quiz_evaluation_prompt, output_key = 'n_review', verbose = True)

# this is the sequential chain which combines the generattion chain and the review chain
generate_evaluate_chain = SequentialChain(
    chains=[quiz_chain, review_chain],  # chains are defined above
    input_variables=["n_text", "n_number", "n_subject", "n_tone", "n_response_json"],  # Original inputs
    output_variables=["n_quiz", "n_review"],  # Unique output keys
    verbose=True,
)

