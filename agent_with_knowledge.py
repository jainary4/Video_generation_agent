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
from agno.models.anthropic import Claude

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
    model=Claude(id="claude-3-7-sonnet-20250219")

    # Create the Agent without the run_id parameter.
    agent = Agent(description="""You are an expert Manim animator specializing in creating 3Blue1Brown-style mathematical animations. Your task is to generate precise, executable Manim Python code for the following mathematical concept.

        # MATHEMATICAL CONCEPT
        <Concept description in LaTeX notation>

        # ANIMATION REQUIREMENTS
        
Create a complete, self-contained Manim scene class that inherits from Scene
Use optimal positioning, scaling, and timing for each element
Ensure all elements are properly visible within the frame
Apply 3Blue1Brown-style color schemes and animation timing
Include appropriate mathematical notation using Tex and MathTex
Utilize VGroups for organizing related objects
Implement smooth transitions between steps
Add camera movements only when they enhance understanding

        # CODE STRUCTURE
        
Implement a main scene class with a clear, descriptive name
Break down the animation into logical steps with comments
Use descriptive variable names for all mobjects
Include proper setup and cleanup for complex animations
Optimize for visual clarity and mathematical insight

        # SPECIFIC CONSIDERATIONS
        
Ensure text is properly sized and positioned relative to other elements
Use appropriate spacing between elements to prevent overlap
Apply consistent color scheme throughout the animation
Ensure animations have appropriate timing (not too fast or slow)
Add wait() calls at key moments to allow viewers to absorb information

        Please generate complete, executable Manim code that visualizes this concept beautifully and precisely.""",
        user_id=user,
        instructions="""Begin by slowly fading in a panoramic star field backdrop to set a cosmic stage. As the camera orients itself to reveal a three-dimensional axis frame, introduce a large title reading 'Quantum Field Theory: 
A Journey into the Electromagnetic Interaction,' written in bold, glowing text at the center of the screen. The title shrinks and moves into the upper-left corner, making room for a rotating wireframe representation of 4D Minkowski spacetime—though rendered in 3D for clarity—complete with a light cone that stretches outward. While this wireframe slowly rotates, bring in color-coded equations of the relativistic metric, such as 
ds2=−c2dt2+dx2+dy2+dz2ds^2 = -c^2 dt^2 + dx^2 + dy^2 + dz^2, with each component highlighted in a different hue to emphasize the negative time component and positive spatial components.

Next, zoom the camera into the wireframe's origin to introduce the basic concept of a quantum field. Show a ghostly overlay of undulating plane waves in red and blue, symbolizing an electric field and a magnetic field respectively, oscillating perpendicularly in sync. Label these fields as E⃗\vec{E} and B⃗\vec{B}, placing them on perpendicular axes with small rotating arrows that illustrate their directions over time. Simultaneously, use a dynamic 3D arrow to demonstrate that the wave propagates along the z-axis. 

As the wave advances, display a short excerpt of Maxwell's equations, morphing from their classical form in vector calculus notation to their elegant, relativistic compact form: ∂μFμν=μ0Jν\partial_\mu F^{\mu \nu} = \mu_0 J^\nu. Animate each transformation by dissolving and reassembling the symbols, underscoring the transition from standard form to four-vector notation.

Then, shift the focus to the Lagrangian density for quantum electrodynamics (QED):
LQED=ψˉ(iγμDμ−m)ψ−14FμνFμν.\mathcal{L}_{\text{QED}} = \bar{\psi}(i \gamma^\mu D_\mu - m)\psi - \tfrac{1}{4}F_{\mu\nu}F^{\mu\nu}.

Project this equation onto a semi-transparent plane hovering in front of the wireframe spacetime, with each symbol color-coded: the Dirac spinor ψ\psi in orange, the covariant derivative DμD_\mu in green, the gamma matrices γμ\gamma^\mu in bright teal, and the field strength tensor FμνF_{\mu\nu} in gold. Let these terms gently pulse to indicate they are dynamic fields in spacetime, not just static quantities. 

While the Lagrangian is on screen, illustrate the gauge invariance by showing a quick animation where ψ\psi acquires a phase factor eiα(x)e^{i \alpha(x)}, while the gauge field transforms accordingly. Arrows and short textual callouts appear around the equation to explain how gauge invariance enforces charge conservation.
Next, pan the camera over to a large black background to present a simplified Feynman diagram. Show two electron lines approaching from the left and right, exchanging a wavy photon line in the center. 

The electron lines are labeled e−e^- in bright blue, and the photon line is labeled γ\gamma in yellow. Subtitles and small pop-up text boxes narrate how this basic vertex encapsulates the electromagnetic interaction between charged fermions, highlighting that the photon is the force carrier. Then, animate the coupling constant α≈1137\alpha \approx \frac{1}{137} flashing above the diagram, gradually evolving from a numeric approximation to the symbolic form α=e24πϵ0ℏc\alpha = \frac{e^2}{4 \pi \epsilon_0 \hbar c}.

Afterward, transition to a 2D graph that plots the running of the coupling constant α\alpha with respect to energy scale, using the renormalization group flow. As the graph materializes, a vertical axis labeled 'Coupling Strength' and a horizontal axis labeled 'Energy Scale' come into view, each sporting major tick marks and numerical values. The curve gently slopes upward, illustrating how α\alpha grows at higher energies, with dynamic markers along the curve to indicate different experimental data points. Meanwhile, short textual captions in the corners clarify that this phenomenon arises from virtual particle-antiparticle pairs contributing to vacuum polarization.

In the final sequence, zoom back out to reveal a cohesive collage of all elements: the rotating spacetime grid, the undulating electromagnetic fields, the QED Lagrangian, and the Feynman diagram floating in the foreground. Fade in an overarching summary text reading 'QED: Unifying Light and Matter Through Gauge Theory,' emphasized by a halo effect. The camera then slowly pulls away, letting the cosmic background re-emerge until each component gracefully dissolves, ending on a single star field reminiscent of the opening shot. A concluding subtitle, 'Finis,' appears, marking the animation's closure and prompting reflection on how fundamental quantum field theory is in describing our universe.""",
        knowledge=knowledge_base,
        reasoning= True,
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
