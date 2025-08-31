import os

def set_environment():

    os.environ["OPENAI_API_KEY"]  = ""
    os.environ["OPENAI_BASE_URL"] = ""
    #os.environ["OPENAI_BASE_URL"] = "https://api.siliconflow.cn/v1"

    os.environ["REPLICATE_API_TOKEN"] = ""
    os.environ["GOOGLE_API_KEY"]=""
 
    os.environ["WOLFRAM_ALPHA_APPID"] = ""

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "My Project"
    os.environ["LANGSMITH_API_KEY"] = ""
    os.environ['LANGSMITH_ENDPOINT'] = ""

    os.environ["OWM_API_KEY"] = ""