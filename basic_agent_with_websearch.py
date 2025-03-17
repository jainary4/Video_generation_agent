from agno.agent import Agent, RunResponse
from agno.models.deepseek import DeepSeek
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.python import PythonTools

coding_agent = Agent(model=DeepSeek(),description="you are a competitive coder that uses the  manim libraray of python to generate videos" ,tools=[PythonTools()], show_tool_calls=True, markdown=True)

# Print the response in the terminal
coding_agent.print_response("give me code that generates a math animation video that explains the concept of derivatives")