import typer
from typing import Optional
from rich.prompt import Prompt

from agno.agent import Agent
from agno.knowledge.json import JSONKnowledgeBase
from agno.vectordb.lancedb import LanceDb
from agno.vectordb.search import SearchType
from agno.document.chunking.semantic import SemanticChunking
from agno.embedder.openai import OpenAIEmbedder
from agno.embedder.ollama import OllamaEmbedder
from agno.models.deepseek import DeepSeek

# Initialize the vector DB with OllamaEmbedder for storing embeddings.
vector_db = LanceDb(
    table_name="manim_docs",
    uri="/tmp/lancedb",
    search_type=SearchType.keyword,
    embedder=OllamaEmbedder()
)

# Create the knowledge base from your JSON file.
# The knowledge base loads the document data, creates embeddings, and stores them in the vector DB.
knowledge_base = JSONKnowledgeBase(
    path="/Users/aryanjain/Downloads/Deep_learning/Startup/manim_docs.json",
    vector_db=vector_db
)

def lancedb_agent(user: str = "user"):
    # Initialize the model that uses the stored embeddings to answer queries.
    model = DeepSeek()

    # Create the Agent without the run_id parameter.
    agent = Agent(
        user_id=user,
        knowledge=knowledge_base,
        model=model,
        show_tool_calls=True,
        debug_mode=True,
    )

    print(f"Agent initialized for user: {user}\n")

    while True:
        message = Prompt.ask(f"[bold] :sunglasses: {user} [/bold]")
        if message.lower() in ("exit", "bye"):
            break
        agent.print_response(message)

if __name__ == "__main__":
    # Load the knowledge base, which processes the JSON file and stores embeddings.
    knowledge_base.load(recreate=True)
    typer.run(lancedb_agent)
