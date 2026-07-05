from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

def ask_dataset_question(df, question, api_key):
    """
    This is the Q&A Agent. It connects to the Gemini API, inspects the 
    cleaned dataframe, and answers natural language questions about it.
    """
    try:
        # Initialize the AI Model (Gemini 1.5 Flash is very fast for data queries)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
        
        # Give the AI access to the specific dataset uploaded by the user
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True,
            allow_dangerous_code=True # Required by LangChain to run Python code locally
        )
        
        # Pass the user's question to the agent and return the answer string
        response = agent.invoke(question)
        return response["output"]
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"