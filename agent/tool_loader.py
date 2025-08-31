#tool_loader.py
"""Utilities for tools."""
from typing import Optional

from langchain import hub
from langchain.agents import AgentExecutor, Tool, create_self_ask_with_search_agent
from langchain.chains import LLMMathChain
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.language_models import BaseLanguageModel
from langchain_core.tools import BaseTool
from langchain_experimental.tools import PythonREPLTool
import os

# Initialize Python REPL tool for code execution
python_repl = PythonREPLTool()

# Configure Wikipedia API wrapper with English language and top 3 results
wikipedia_api_wrapper = WikipediaAPIWrapper(lang="en", top_k_results=3)


def load_tools(
    tool_names: list[str],
    llm: Optional[BaseLanguageModel] = None,
) -> list[BaseTool]:
    """Load and configure tools based on requested tool names.
    
    Args:
        tool_names: List of tool names to load
        llm: Language model for tools that require it (e.g., math calculations)
        
    Returns:
        list[BaseTool]: List of configured tool instances
    """
    
    # Try to create DuckDuckGo search tool first using the newer ddgs package
    ddg_tool = None
    try:
        from ddgs import DDGS
        
        # Custom DuckDuckGo search implementation with error handling
        class CustomDuckDuckGoSearch:
            def search(self, query: str) -> str:
                """Search the web using DuckDuckGo and format results.
                
                Args:
                    query: Search query string
                    
                Returns:
                    str: Formatted search results or error message
                """
                try:
                    # Perform search with DDGS context manager
                    with DDGS() as ddgs:
                        results = ddgs.text(query, max_results=5)
                        formatted_results = []
                        
                        # Format each search result
                        for r in results:
                            title = r.get('title', 'No title')
                            body = r.get('body', 'No description')
                            formatted_results.append(f"{title}: {body}")
                            
                        return "\n".join(formatted_results)
                except Exception as e:
                    # Return error message if search fails
                    return f"Search failed: {str(e)}"
        
        # Create DuckDuckGo search tool
        ddg_tool = Tool(
            name="ddg-search",
            description="Search the web using DuckDuckGo. Input should be a search query.",
            func=CustomDuckDuckGoSearch().search,
        )
    except ImportError:
        # Fallback warning if ddgs package is not installed
        print("Warning: ddgs package not installed. Please install it with `pip install ddgs`")
        
        # Fallback to old implementation if new package not available
        try:
            from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
            from langchain_community.utilities.duckduckgo_search import DuckDuckGoSearchAPIWrapper
            import warnings
            
            # Suppress warnings for deprecated components
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ddg_tool = DuckDuckGoSearchRun(
                    api_wrapper=DuckDuckGoSearchAPIWrapper(
                        region="wt-wt",
                        safesearch="moderate"
                    )
                )
        except Exception as e:
            print(f"Warning: Could not initialize DuckDuckGo search tool: {e}")

    # Load prompt for self-ask with search agent
    prompt = hub.pull("hwchase17/self-ask-with-search")
    
    # Create search tool for self-ask agent
    search_wrapper = ddg_tool if ddg_tool else None
    if search_wrapper:
        # Use actual search tool if available
        search_tool = Tool(
            name="Intermediate Answer",
            func=search_wrapper.invoke,
            description="Search",
        )
    else:
        # Create a dummy search tool if DDG is not available
        search_tool = Tool(
            name="Intermediate Answer",
            func=lambda x: "Search tool not available",
            description="Search",
        )
    
    # Create self-ask agent for handling complex questions that require search
    self_ask_agent = AgentExecutor(
        agent=create_self_ask_with_search_agent(
            llm=llm,
            tools=[search_tool],
            prompt=prompt,
        ),
        tools=[search_tool],
        handle_parsing_errors=True,
    )

    # Build available tools dictionary dynamically
    available_tools = {
        "arxiv": ArxivQueryRun(api_wrapper=ArxivAPIWrapper()),
        "wikipedia": WikipediaQueryRun(api_wrapper=wikipedia_api_wrapper),
        "python_repl": Tool(
            name="python_repl",
            description="A Python shell. Use this to execute python commands."
            " Input should be a valid python command. If you want to see"
            " the output of a value, you should print it out with `print(...)`. ",
            func=python_repl.run,
        ),
        "llm-math": Tool(
            name="Calculator",
            description="Useful for when you need to answer questions about math.",
            func=LLMMathChain.from_llm(llm=llm).run,
            coroutine=LLMMathChain.from_llm(llm=llm).arun,
        ),
        "critical_search": Tool.from_function(
            func=self_ask_agent.invoke,
            name="Self-ask agent",
            description="A tool to answer complicated questions. "
            "Useful for when you need to answer questions about current events. "
            "Input should be a question.",
        ),
    }

    # Add DuckDuckGo search tool if available
    if ddg_tool:
        available_tools["ddg-search"] = ddg_tool

    # Try to add OpenWeatherMap tool if credentials are available
    if os.environ.get("OWM_API_KEY"):
        try:
            from pyowm import OWM
            
            # Custom OpenWeatherMap tool implementation
            class OpenWeatherMapTool:
                def __init__(self, api_key: str):
                    """Initialize OpenWeatherMap tool with API key.
                    
                    Args:
                        api_key: OpenWeatherMap API key from environment variables
                    """
                    self.owm = OWM(api_key)
                    self.mgr = self.owm.weather_manager()
                
                def get_current_weather(self, location: str) -> str:
                    """Get current weather information for a specific location.
                    
                    Args:
                        location: Location name or city name
                        
                    Returns:
                        str: Formatted weather information or error message
                    """
                    try:
                        # Get weather observation for the location
                        observation = self.mgr.weather_at_place(location)
                        weather = observation.weather
                        
                        # Extract weather data
                        temp = weather.temperature('celsius')['temp']
                        status = weather.detailed_status
                        humidity = weather.humidity
                        wind_speed = weather.wind()['speed']
                        
                        # Format and return weather information
                        return f"Current weather in {location}: {temp}°C, {status}, Humidity: {humidity}%, Wind Speed: {wind_speed} m/s"
                    except Exception as e:
                        # Return error message if weather data cannot be retrieved
                        return f"Could not retrieve weather data for {location}: {str(e)}"
            
            # Create a proper Langchain Tool from our custom class
            owm_tool = OpenWeatherMapTool(os.environ["OWM_API_KEY"])
            
            # Add OpenWeatherMap tool to available tools
            available_tools["openweathermap"] = Tool(
                name="OpenWeatherMap",
                description="Get current weather information for a specific location. Use this when asked about current weather conditions. Input should be a location name or a city name like 'Beijing' or 'New York'.",
                func=owm_tool.get_current_weather,
            )
        except ImportError:
            print("Warning: pyowm package not installed. Please install it with `pip install pyowm`")
        except Exception as e:
            print(f"Warning: Could not initialize OpenWeatherMap tool: {e}")
    else:
        print("Info: OWM_API_KEY not found. OpenWeatherMap tool will not be available.")
    
    # Try to add Wolfram Alpha tool if credentials and dependencies are available
    if os.environ.get("WOLFRAM_ALPHA_APPID"):
        try:
            from langchain_community.tools.wolfram_alpha import WolframAlphaQueryRun
            from langchain_community.utilities.wolfram_alpha import WolframAlphaAPIWrapper
            
            # Create a wrapper with error handling
            class WolframAlphaTool(WolframAlphaQueryRun):
                def _run(self, query: str) -> str:
                    """Execute Wolfram Alpha query with error handling.
                    
                    Args:
                        query: Query string for Wolfram Alpha
                        
                    Returns:
                        str: Query results or error message
                    """
                    try:
                        return super()._run(query)
                    except Exception as e:
                        return f"Wolfram Alpha query failed: {str(e)}. Please try a different tool or rephrase your query."
                
                async def _arun(self, query: str) -> str:
                    """Execute async Wolfram Alpha query with error handling.
                    
                    Args:
                        query: Query string for Wolfram Alpha
                        
                    Returns:
                        str: Query results or error message
                    """
                    try:
                        return await super()._arun(query)
                    except Exception as e:
                        return f"Wolfram Alpha query failed: {str(e)}. Please try a different tool or rephrase your query."
            
            # Add Wolfram Alpha tool to available tools
            available_tools["wolfram-alpha"] = WolframAlphaTool(api_wrapper=WolframAlphaAPIWrapper())
        except ImportError:
            print("Warning: wolframalpha package not installed. Please install it with `pip install wolframalpha`")
        except Exception as e:
            print(f"Warning: Could not initialize Wolfram Alpha tool: {e}")
    else:
        print("Info: WOLFRAM_ALPHA_APPID not found. Wolfram Alpha tool will not be available.")

    # Try to add Google Search tool if credentials are available
    if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GOOGLE_CSE_ID"):
        try:
            # Using the newer imports to avoid deprecation warnings
            from langchain_google_community import GoogleSearchAPIWrapper, GoogleSearchRun
            
            # Add Google Search tool using newer implementation
            available_tools["google-search"] = GoogleSearchRun(
                api_wrapper=GoogleSearchAPIWrapper(
                    k=5,
                ),
                search_params={"dateRestrict": "m12"}
            )
        except ImportError:
            print("Warning: langchain-google-community package not installed. Please install it with `pip install -U langchain-google-community`")
            
            # Fallback to deprecated versions with warning suppression
            try:
                from langchain_community.utilities.google_search import GoogleSearchAPIWrapper
                from langchain_community.tools.google_search import GoogleSearchRun
                import warnings
                
                # Suppress warnings for deprecated components
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    available_tools["google-search"] = GoogleSearchRun(
                        api_wrapper=GoogleSearchAPIWrapper(
                            k=5,
                        ),
                        search_params={"dateRestrict": "m12"}
                    )
            except Exception as e:
                print(f"Warning: Could not initialize Google Search tool: {e}")
        except Exception as e:
            print(f"Warning: Could not initialize Google Search tool: {e}")
    else:
        print("Info: GOOGLE_API_KEY and/or GOOGLE_CSE_ID not found. Google Search tool will not be available.")
    
    # Build list of requested tools
    tools = []
    for name in tool_names:
        if name in available_tools:
            tools.append(available_tools[name])
        else:
            print(f"Warning: Tool '{name}' not found in available tools.")
    
    return tools