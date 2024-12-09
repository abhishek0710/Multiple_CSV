import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import ChatOpenAI, OpenAI
import io

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
st.title("Chat with your excel data")

def generate_plot(df, plot_type, x_column, y_column, title):
    plt.figure(figsize=(10, 6))
    if plot_type == 'bar':
        df.plot(kind='bar', x=x_column, y=y_column)
    elif plot_type == 'line':
        df.plot(kind='line', x=x_column, y=y_column)
    elif plot_type == 'scatter':
        plt.scatter(df[x_column], df[y_column])
    plt.title(title)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.tight_layout()
    
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

# File Upload
uploaded_file = st.file_uploader("Upload CSV/Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Determine the file type and read accordingly
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        df = pd.read_excel(uploaded_file)
    
    # Display top 10 rows of the DataFrame
    st.subheader("Top 10 Rows of the DataFrame:")
    st.write(df.head(10))
    
    # Save the uploaded file to a temporary location
    temp_file_path = f"temp_{uploaded_file.name}"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Create the CSV agent with dangerous code execution allowed
    agent = create_csv_agent(
        path=temp_file_path,
        llm=ChatOpenAI(api_key=OPENAI_API_KEY, temperature=0, model="gpt-3.5-turbo"),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True  # Opt-in to allow dangerous code execution
    )
    
    message = st.text_input("Enter the prompt (or 'exit' to leave): ")
    if message:
        if message.lower() == 'exit':
            st.stop()
        response = agent.run(message)
        st.write(response)
        
        # Check if the response contains instructions for plotting
        if "PLOT:" in response:
            plot_instructions = response.split("PLOT:")[1].strip()
            try:
                plot_type, x_column, y_column, title = plot_instructions.split(',')
                plot_buf = generate_plot(df, plot_type.strip(), x_column.strip(), y_column.strip(), title.strip())
                st.image(plot_buf)
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
    
    # Add a section for manual plot creation
    st.subheader("Create a Plot")
    plot_type = st.selectbox("Select plot type", ['bar', 'line', 'scatter'])
    x_column = st.selectbox("Select X-axis column", df.columns)
    y_column = st.selectbox("Select Y-axis column", df.columns)
    plot_title = st.text_input("Enter plot title")
    
    if st.button("Generate Plot"):
        plot_buf = generate_plot(df, plot_type, x_column, y_column, plot_title)
        st.image(plot_buf)

else:
    st.write("Please upload a file to proceed.")
