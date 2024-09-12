import json
import streamlit as st
from streamlit_chat import message
from dotenv import load_dotenv
from streamlit_lottie import st_lottie_spinner
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate
)
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL
from streamlit_cookies_controller import CookieController

def render_animation():
    path = "assets/typing_animation.json"
    with open(path,"r") as file: 
        animation_json = json.load(file) 
        return animation_json

def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):
        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

st.set_page_config(
    page_title="Softsquare AI",
    page_icon="🤖",
)

load_dotenv()
openaiModels = st.secrets["OPENAI_MODEL"]
portKeyApi = st.secrets["PORTKEY_API_KEY"]
pinecone_index = st.secrets["PINECONE_INDEX_NAME"]

# Load Animation
typing_animation_json = render_animation()
hide_st_style = """ <style>
                    #MainMenu {visibility:hidden;}
                    footer {visibility:hidden;}
                    header {visibility:hidden;}
                    </style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)
st.markdown("""
    <h1 id="chat-header" style="position: fixed;
                   top: 0;
                   left: 0;
                   width: 100%;
                   text-align: center;
                   background-color: #f1f1f1;
                   z-index: 9">
        Chat with Salesforce AI Bot
    </h1>
""", unsafe_allow_html=True)

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Hi there, I am your Salesforce Assist. How can I help you today?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'initialPageLoad' not in st.session_state:
    st.session_state['initialPageLoad'] = False

if 'prevent_loading' not in st.session_state:
    st.session_state['prevent_loading'] = False

if 'email' not in st.session_state:
    st.session_state['email'] = ''

embeddings = OpenAIEmbeddings()
controller = CookieController()

# with st.sidebar:
#     emailInput = st.text_input("Enter Your Email")
#     if emailInput != '' and emailInput != None:
#         controller.set("email_id",emailInput)

email_id = str(controller.get('email_id'))
user_id = controller.get("ajs_anonymous_id")

st.session_state.email = email_id


if email_id != '' and email_id != None and email_id != 'None':
    st.markdown("""
    <style>
        section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)

portkey_headers = createHeaders(api_key=portKeyApi,provider="openai", metadata={"email_id": email_id, "_user_id" :user_id } )

llm = ChatOpenAI(temperature=0,
                model=openaiModels,
                base_url=PORTKEY_GATEWAY_URL,
                default_headers=portkey_headers
               )

vector_store = PineconeVectorStore(index_name=pinecone_index, embedding=embeddings)

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferMemory(memory_key="chat_history",
                                    max_len=50,
                                    return_messages=True,
                                    output_key='answer')

# Answer the question as truthfully as possible using the provided context, 
# and if the answer is not contained within the text below, say 'I don't know'
general_system_template = r""" 
    You are an AI assistant specialized in Salesforce Sales Cloud and Service Cloud, designed to support Salesforce admins, developers, business analysts, and consultants. Your role is to analyze business problem statements, provide feature explanations, offer customization options, and deliver step-by-step implementation guides. The solutions you provide should align with Salesforce’s best practices and ensure the appropriate fit for Sales or Service Cloud.
Key Objectives:
    Understand User Queries: Use Natural Language Processing (NLP) to accurately interpret complex business problems and questions.
    Persona Identification: Recognize the user's role (Admin, Developer, Business Analyst, Consultant) and tailor responses to meet their specific needs and technical expertise.
Knowledge Base Integration:
    Salesforce Documentation: Utilize the official Salesforce Sales Cloud and Service Cloud documentation, including user manuals, object models, feature explanations, and architecture overviews.
    Public Salesforce Knowledge: In cases where official documentation doesn’t fully address the user’s needs, leverage publicly available Salesforce knowledge resources (e.g., help articles, community forums, blogs) to provide more comprehensive answers.
    Feature Matching: Analyze the user’s problem statement and map it to Salesforce features, functionality, and cloud solutions (Sales or Service).
    Customization & Configuration: Offer detailed guidance on customization options and implementation steps that suit the user's problem, ensuring solutions are aligned with Salesforce best practices.
Contextual Clarification:
    Follow-Up Questions: Ask clarifying questions if the user’s query is unclear. Ensure full context is understood before offering a solution to improve accuracy and relevance.
Conversation Analysis:
    Key Attribute Extraction: Identify important attributes such as objects, features, and relationships in the user query to deliver a precise response.
Solution Approach:
    Step-by-Step Instructions: Provide clear, step-by-step guidance for configuring features like lead scoring, opportunity pipeline reporting, and other Salesforce customizations.
    Feature Comparison: If more than one solution applies, compare available features to recommend the optimal approach for Sales or Service Cloud.
Solution Context:
    Cloud-Specific Solutions: Ensure your responses indicate whether the problem is best handled within Sales Cloud or Service Cloud, providing clarity on which cloud fits the user’s needs.
Troubleshooting:
    Common Issues & Solutions: Provide troubleshooting steps based on Salesforce's documented best practices for common issues in Sales and Service Cloud.
    Public Knowledge and Escalation:
        Leverage Public Salesforce Resources: When necessary, look up publicly available Salesforce resources such as articles from Salesforce Help, Trailhead, or other credible Salesforce communities to fill knowledge gaps or enhance your answers.
    Escalation Path: For complex issues that can’t be resolved using the available documentation, guide users on how to escalate the issue to Salesforce support or involve a consultant.
    Example Queries:
        1. "How do I configure lead scoring?"
                Determine if the question fits Sales Cloud.
                Provide step-by-step instructions for configuring lead scoring.
                Supplement with additional tips from public Salesforce resources if required.

        2. "How can I create an opportunity pipeline report?"
                Identify the reporting needs.
                Guide the user through the process of creating the report, ensuring they understand the objects and fields involved.
                Include links to related knowledge articles if needed.
DOs:
    Personalize responses according to the user’s role and expertise.
    Emphasize Salesforce's scalability and adaptability when suggesting solutions.
    Reference Salesforce’s official documentation and credible public resources to validate your answers.
DON’Ts:
    Avoid overcomplicating responses; aim for clarity and ease of understanding.
    Steer clear of excessive jargon unless addressing technical personas (e.g., developers).
    Response Style:
    Deliver responses in a human-like, conversational tone.
    Use bullet points and short paragraphs to enhance readability.
----
{context}
----
"""
general_user_template = "Question:```{question}```"

system_msg_template = SystemMessagePromptTemplate.from_template(template=general_system_template)

human_msg_template = HumanMessagePromptTemplate.from_template(template=general_user_template)
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]
qa_prompt = ChatPromptTemplate.from_messages( messages )
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    vector_store.as_retriever(search_kwargs={'k': 2}),
    verbose=True,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    rephrase_question = True,
    response_if_no_docs_found = "Sorry, I dont know",
    memory = st.session_state.buffer_memory,
    
)

# container for chat history
response_container = st.container()
textcontainer = st.container()


chat_history = []
with textcontainer:
    st.session_state.initialPageLoad = False
    query = st.chat_input(placeholder="Say something ... ", key="input")
    if query and query != "Menu":
        conversation_string = get_conversation_string()
        with st_lottie_spinner(typing_animation_json, height=50, width=50, speed=3, reverse=True):
            response = qa_chain({'question': query, 'chat_history': chat_history})
            chat_history.append((query, response['answer']))
            # print("response:::: ",response)
            st.session_state.requests.append(query)
            st.session_state.responses.append(response['answer'])
    st.session_state.prevent_loading = True



with response_container:
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.session_state.initialPageLoad = False
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            response = f"<div style='font-size:0.875rem;line-height:1.75;white-space:normal;'>{st.session_state['responses'][i]}</div>"
            message(response,allow_html=True,key=str(i),logo=('https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/SS512X512.png'))
            if i < len(st.session_state['requests']):
                request = f"<meta name='viewport' content='width=device-width, initial-scale=1.0'><div style='font-size:.875rem'>{st.session_state['requests'][i]}</div>"
                message(request, allow_html=True,is_user=True,key=str(i)+ '_user',logo='https://raw.githubusercontent.com/Maniyuvi/SoftsquareChatbot/main/generic-user-icon-13.jpg')


