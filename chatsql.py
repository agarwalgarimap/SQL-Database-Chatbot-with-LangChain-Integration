import streamlit as st
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_sql_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
import os


# Set up Streamlit page configuration
st.set_page_config(page_title="LangChain: Chat with MySQL DB", page_icon="🦜")
# App title
st.title("🦜 LangChain: Chat with SQL DB")

# Add larger font size for the subtitle
st.markdown(
    "<h2 style='text-align: left; font-size: 25px;'>How can I help you?</h2>",
    unsafe_allow_html=True
)

# Sidebar: MySQL Database Configuration
st.sidebar.header("MySQL Database Configuration")
mysql_host = st.sidebar.text_input("MySQL Host:")
mysql_user = st.sidebar.text_input("MySQL User:")
mysql_password = st.sidebar.text_input("MySQL Password:", type="password")
mysql_db = st.sidebar.text_input("MySQL Database:")

# API Key Input
openai_api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
if not openai_api_key:
    st.warning("Please provide an OpenAI API key to proceed.")
    st.stop()

# Function to configure the MySQL database
@st.cache_resource(ttl="2h")
def configure_mysql_db(host, user, password, db_name):
    if not (host and user and password and db_name):
        st.error("Please provide all MySQL connection details.")
        st.stop()
    return SQLDatabase.from_uri(f"mysql+pymysql://{user}:{password}@{host}/{db_name}")

# Configure the MySQL database
db = configure_mysql_db(mysql_host, mysql_user, mysql_password, mysql_db)

# Initialize LangChain Toolkit with OpenAI API
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-4o-mini", streaming=True)

# Initialize Streamlit container for callbacks
callback_container = st.empty()

# Initialize the toolkit and create the agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    callbacks=[StreamlitCallbackHandler(parent_container=callback_container)]
)

# Query input section
query = st.text_area("", height=200)

# Button to trigger query execution
if st.button("Run Query"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner("Processing..."):
                response = agent_executor.run(query)
            st.success("Query executed successfully!")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")