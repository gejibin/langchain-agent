
# LangChain Agent

## Description
The LangChain Agent is a versatile tool designed to answer research questions using various reasoning strategies and multiple AI models. The system integrates with numerous external tools and APIs, allowing it to perform complex tasks such as mathematical computations, web searches, weather information retrieval, and more.

Key features include:
- **Multiple Reasoning Strategies**: Supports "zero-shot-react" and "plan-and-solve" strategies.
- **Diverse AI Models**: Compatible with models like `gpt-3.5-turbo`, `gpt-4`, `Qwen/Qwen3-8B`, and `Qwen/Qwen2.5-7B`.
- **Extensive Tool Integration**: Utilizes tools such as DuckDuckGo search, Wikipedia, Arxiv, Wolfram Alpha, Google Search, OpenWeatherMap, and Python REPL.
- **Interactive Streamlit Interface**: Provides an interactive web interface built with Streamlit for user-friendly interaction.

## Installation

### Prerequisites
- Python 3.8 or higher
- Access to API keys for OpenAI, Google, Wolfram Alpha, and OpenWeatherMap (optional but recommended for full functionality)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/langchain-agent.git
   cd langchain-agent
   ```
2. **Set Up Virtual Environment (Optional but Recommended)**
   ```bash
   pip install poetry
   poetry install
   ```
3. **Configure Environment Variables**

   Ensure that the following variables are set correctly in the [set_environment()](file://d:\code\langchain-agent\agent\config.py#L2-L25) function within [config.py](file://d:\code\langchain-agent\agent\config.py):
   ```python
   os.environ["OPENAI_API_KEY"] = "your_openai_api_key"
   os.environ["OPENAI_BASE_URL"] = "your_openai_base_url"
   os.environ["REPLICATE_API_TOKEN"] = "your_replicate_api_token"
   os.environ["GOOGLE_API_KEY"] = "your_google_api_key"
   os.environ["WOLFRAM_ALPHA_APPID"] = "your_wolfram_alpha_appid"
   os.environ["OWM_API_KEY"] = "your_openweathermap_api_key"
   os.environ["LANGCHAIN_TRACING_V2"] = "true"
   os.environ["LANGCHAIN_PROJECT"] = "My Project"
   os.environ["LANGSMITH_API_KEY"] = "your_langsmith_api_key"
   os.environ['LANGSMITH_ENDPOINT'] = "https://api.smith.langchain.com"
   ```
   Replace the placeholder values with your actual API keys and credentials.

4. **Run the Application**
   ```bash
   poetry shell (or poetry env activate) # activates virtual environment

   streamlit run app/app.py
   ```

## Usage
Once the application is running, you can interact with it via the Streamlit interface:
- Select a reasoning strategy (`zero-shot-react` or `plan-and-solve`).
- Choose an AI model from the dropdown menu.
- Select the tools you wish to use for your query.
- Input your question and receive responses based on the selected strategy and tools.

## Features

### Reasoning Strategies
- **Zero-Shot React**: A strategy that uses zero-shot learning to react to queries directly.
- **Plan-and-Solve**: A more structured approach where the agent plans its steps before solving the problem.

### Tools
- **DuckDuckGo Search**: Perform web searches to gather information.
- **Wikipedia**: Retrieve information from Wikipedia articles.
- **Arxiv**: Access academic papers from Arxiv.
- **Wolfram Alpha**: Perform complex calculations and retrieve factual data.
- **Google Search**: Perform Google searches with specific parameters.
- **OpenWeatherMap**: Get current weather information for specified locations.
- **Python REPL**: Execute Python commands directly.

### Memory and Conversation Context
The application maintains conversation history using `ConversationBufferMemory`, ensuring contextual continuity across interactions.

## Contributing
Contributions are welcome! Please ensure that any new features or bug fixes are well-documented and include appropriate tests.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
```

