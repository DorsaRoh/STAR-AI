
# Import necessary libraries
import os 
import openai

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from pdfreader import PDFDocument, SimplePDFViewer



# Set Streamlit page configuration & LOGO
from PIL import Image
# Loading Image using PIL
im = Image.open('content/logo.png')
# Adding Image to web app
st.set_page_config(page_title='S.T.A.R.', layout='wide', page_icon = im)


# save user uplaoded files to /data folder
def fileSaver():
    uploaded_file = st.sidebar.file_uploader("Upload mission data", type='.pdf')

    if uploaded_file is not None and valid_apikey():
        with open(os.path.join("data", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File has been saved successfully!")

        # create a text file with the pdf's text content
        file_path = os.path.join("data", uploaded_file.name)
        text_content = extract_text_from_pdf(file_path)
        with open(os.path.splitext(file_path)[0]+".txt", "w") as f: 
            f.write(text_content)




# Side bar

# Side bar api key
placeholder_text_prompt = "Please enter your key"
openai_api_key = st.sidebar.text_input("OpenAI API Key",value="", help="", key="api_input", placeholder=placeholder_text_prompt)

os.environ['OPENAI_API_KEY'] = openai_api_key


st.sidebar.title(":blue[Revolutionizing Technical Standards with AI]")
st.sidebar.markdown("To use AI to help mission designers streamline the process of sifting through technical requirements, detecting omissions, inconsistencies, and offering requirement recommendations.")


fileSaver()
st.write("#")

# Application details
st.sidebar.subheader(":blue[1. [Data-Driven Decisions](https://github.com/DorsaRoh/Custom-AI/tree/main/RealizeAI)]")
st.sidebar.markdown("*Learns from the successes and failures of 50 past space missions.*")

st.sidebar.subheader(":blue[2. [Real-time Feedback](https://github.com/DorsaRoh/Custom-AI/tree/main/PatientGPT.AI)]")
st.sidebar.markdown("*Input your mission parameters and receive immediate feedback and insights.*")

st.sidebar.subheader(":blue[3. [Comprehensive Knowledge Base](https://github.com/DorsaRoh/Custom-AI/tree/main/PatientGPT.AI)]")
st.sidebar.markdown("*Built on data from leading space agencies and research organizations.*")



# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False


# APP LAYOUT
# Title
st.title('STAR: AI Co-Pilot for Mission *Success*')
st.subheader(':blue[Streamline your mission design with the power of AI-driven insights.]')
st.markdown("___")

#columns for layout
col1, col2 = st.columns(2)


 #If invalid/no api key enteblue, show warning
def valid_apikey():
    if openai_api_key.startswith('sk-'):
        return True
    else:
        st.warning('Invalid API Key', icon='⚠')
        return False
    
if valid_apikey():
    col2.markdown(":blue[AI Response]")

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False


# Langchain LLM that feeds off of user data
def load_model():
    if PERSIST and os.path.exists("persist"):
        st.write("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = DirectoryLoader("data/")
        if PERSIST:
            index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    )

    return chain

chain = load_model()
chat_history = []



# FILE SAVE TO DATA FOLDER

# Extract text from user uploaded pdf file - so LLM can read it
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as fd:
        viewer = SimplePDFViewer(fd)
        text = ""
        for page in viewer:
            viewer.render()
            text += ' '.join(viewer.canvas.strings)
    return text




# Prompt templates for Langchain
class PromptTemplate:
    def __init__(self, input_variables, template):
        self.input_variables = input_variables
        self.template = template

    def format(self, **kwargs):
        # Check if all necessary variables have been provided
        for variable in self.input_variables:
            if variable not in kwargs:
                raise ValueError(f"Missing input variable: {variable}")

        # Use the provided variables to format the template
        return self.template.format(**kwargs)

title_template = PromptTemplate(
    input_variables=['topic'], 
    template='space mission'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'], 
    template=(
        "streamline the process of sifting through technical requirements, detecting omissions, inconsistencies, and offering requirement recommendations using {title}. Leverage {wikipedia_research}"
    )
)




# ADDITIONAL QUESTIONS

def generate_questions_response(input_text):
    chain = load_model()
    chat_history = []

    query = input_text

    if query not in ['', 'quit', 'q', 'exit']:
        result = chain({"question": query, "chat_history": chat_history})
        st.write(f"AI: {result['answer']}")
        chat_history.append((query, result['answer']))

#LAYOUT FOR COLUMN 1
with col1:
    chain = load_model()


    
    css = '''
    <style>
        .element-container:has(>.stTextArea), .stTextArea {
            width: 100% !important;
            max-width: 800px;
        }

        .stTextArea textarea {
            height: 400px;
            width: 100% !important;
        }

        @media (max-width: 800px) {
            .stTextArea textarea {
                height: 300px;
            }
        }
    </style>
    '''

    #response = st.text_area("Type here")
    #st.write(response)
    #st.write(css, unsafe_allow_html=True)

    
    placeholder_text_prompt = '''
    Mission Case: Operation Lunar Scribe

    Objective: To deploy a rover on the Moon's southern pole and collect soil samples to analyze water ice concentrations, which will be transmitted back to Earth.

    Launch Vehicle:

    Name: FalconX Prime
    Type: Reusable Heavy-Lift Launch Vehicle
    Propellant: Liquid Hydrogen (LH2) and Liquid Oxygen (LOX)
    Fault: The launch window provided is during a heavy meteor shower, which poses significant risks.
    Rover:

    Name: Scribe-1
    Weight: 850 kg
    Dimensions: 2.5m x 2m x 1.5m
    Power Source: Solar Panels
    Fault: Solar Panels are not optimal for the lunar southern pole, which has prolonged darkness. A nuclear battery would be more suitable.
    Landing Site:

    Location: Malapert Mountain, Lunar South Pole
    Terrain: Rocky with ice patches
    Fault: The selected landing site is close to a known deep crevasse, which can jeopardize the rover's safety.
    Communication:

    Method: Direct Line of Sight with Earth
    Frequency: 450 MHz
    Fault: The rover's communication system does not account for the times when the Moon's southern pole is not facing Earth, leading to potential long communication blackouts.
    Rover Mobility:

    Wheel System: Standard rubber tires
    Fault: Rubber tires are not suitable for lunar terrain; they should be made of a metal alloy or a specialized material to handle the rocky and icy terrain.
    Sample Collection:

    Method: Drill and Scoop
    Storage: Onboard sealed containers
    Fault: The sample storage containers are not vacuum-sealed, which may lead to contamination of the lunar samples.
    Mission Duration:

    Rover Operational Time: 30 Earth days
    Fault: Given the power source, rover design, and the harsh conditions of the lunar pole, the operational time is overly optimistic.
    Return Strategy:

    There is no provision for returning the samples back to Earth. Only digital data will be sent.
    Fault: The primary objective includes physically analyzing samples on Earth, but there's no provision for returning the samples.
    Budget:

    Allocated: $250 million
    Fault: Given the complexity of the mission and the requirements, the budget seems insufficient.

    '''

    # script = st.text_area("Enter mission requirements:",value="", help="", key="prompt_input", placeholder=placeholder_text_prompt)
    st.write(css, unsafe_allow_html=True)
    

    with st.form('additional_questions_form'):
        placeholder_text_additional = '''
        Mission Case: Operation Lunar Scribe

        Objective: To deploy a rover on the Moon's southern pole and collect soil samples to analyze water ice concentrations, which will be transmitted back to Earth.


        Launch Vehicle: FalconX Prime
        Type: Reusable Heavy-Lift Launch Vehicle
        Propellant: Liquid Hydrogen (LH2) and Liquid Oxygen (LOX)
        Name: Scribe-1
        Weight: 850 kg
        Dimensions: 2.5m x 2m x 1.5m
        Power Source: Solar Panels
        Fault: Solar Panels are not optimal for the lunar southern pole, which has prolonged darkness. A nuclear battery would be more suitable.

        Location: Malapert Mountain, Lunar South Pole
        Terrain: Rocky with ice patches
        Fault: The selected landing site is close to a known deep crevasse, which can jeopardize the rover's safety.
        Communication:

        Method: Direct Line of Sight with Earth
        Frequency: 450 MHz
        Fault: The rover's communication system does not account for the times when the Moon's southern pole is not facing Earth, leading to potential long communication blackouts.
        Rover Mobility:

        Wheel System: Standard rubber tires
        Fault: Rubber tires are not suitable for lunar terrain; they should be made of a metal alloy or a specialized material to handle the rocky and icy terrain.
        Sample Collection:

        Method: Drill and Scoop
        Storage: Onboard sealed containers
        Fault: The sample storage containers are not vacuum-sealed, which may lead to contamination of the lunar samples.
        Mission Duration:

        Rover Operational Time: 30 Earth days
        Fault: Given the power source, rover design, and the harsh conditions of the lunar pole, the operational time is overly optimistic.
        Return Strategy:

        There is no provision for returning the samples back to Earth. Only digital data will be sent.
        Fault: The primary objective includes physically analyzing samples on Earth, but there's no provision for returning the samples.
        Budget:

        Allocated: $250 million
        Fault: Given the complexity of the mission and the requirements, the budget seems insufficient.
        '''
        query = st.text_area('Enter mission details:',value="", help="", key="additional_input", placeholder=placeholder_text_additional)    
        submitted = st.form_submit_button(label='Submit')
        valid_apikey()
        if submitted and valid_apikey():
            with col2:
                generate_questions_response(query)

    

wiki = WikipediaAPIWrapper()


if submitted and valid_apikey():
    script = "space mission failure analysis"
    try:
        wiki = WikipediaAPIWrapper()
        wiki_research = wiki.run(script) 
    
        title_prompt = title_template.format(topic=script)
        title_result = chain({"question": title_prompt, "chat_history": chat_history})
        chat_history.append((title_prompt, title_result['answer']))

        script_prompt = script_template.format(title=script, wikipedia_research=wiki_research)

        
        script_result = chain({"question": script_prompt, "chat_history": chat_history})
        with col2:
            st.write(f"AI: {script_result['answer']}")
            
            chat_history.append((script_prompt, script_result['answer']))
    except TypeError as e:
        st.write("An error occurred: ", e)



# Hide Streamlit's default footer
st.markdown('<style>footer{visibility:hidden;}</style>', unsafe_allow_html=True)