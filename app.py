import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from crewai import Agent, Task, Crew
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize session state
if 'script_result' not in st.session_state:
    st.session_state.script_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
    
# setting up the page header here.
hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

# setting up the page config here.
st.set_page_config(
    page_title="Movie Script Summarizer",
    page_icon="🎞️",
    menu_items={
        'Get Help': 'https://www.linkedin.com/in/abhiiiman',
        'Report a bug': "https://www.github.com/abhiiiman",
        'About': "## Movie Script Summarizer"
    }
)

# removing all the default streamlit configs here
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define LLM Model (Llama3-70B)
llm = ChatGroq(
    temperature=0,
    model_name="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)

# Define agents for movie script generation

# 1. Plot Designer
plot_designer = Agent(
    llm=llm,
    role="Plot Designer",
    goal="Design an engaging and cohesive plot for the movie {movie_name} with the new ending: {desired_ending}",
    backstory="You are tasked with altering the plot of the movie: {movie_name}. The user wants a new ending: {desired_ending}. You need to create a structured plot outline, leading logically to the new ending while staying true to the original movie.",
    allow_delegation=False,
    verbose=True
)

# 2. Character Analyst
character_analyst = Agent(
    llm=llm,
    role="Character Analyst",
    goal="Analyze the main character arcs in {movie_name} and how they evolve with the new ending: {desired_ending}",
    backstory="You are responsible for analyzing the character arcs in {movie_name} and explaining how their journey evolves in light of the new ending: {desired_ending}. Focus on how each main character is impacted by the new plot and their final resolution.",
    allow_delegation=False,
    verbose=True
)

# 3. Setting Designer
setting_designer = Agent(
    llm=llm,
    role="Setting Designer",
    goal="Describe key settings in the movie {movie_name} and how they are influenced by the new ending: {desired_ending}",
    backstory="You are responsible for describing the primary settings of {movie_name}, focusing on how the atmosphere and locations evolve as the plot shifts toward the new ending: {desired_ending}.",
    allow_delegation=False,
    verbose=True
)

# 4. Theme Analyst
theme_analyst = Agent(
    llm=llm,
    role="Theme Analyst",
    goal="Analyze the themes in {movie_name} and how the new ending: {desired_ending} impacts these themes.",
    backstory="Your task is to analyze the themes and motifs of the movie {movie_name}. You must explain how the original themes are influenced by the new plot and ending, and describe any new themes that arise due to the altered storyline.",
    allow_delegation=False,
    verbose=True
)

# 5. Script Summarizer
script_summarizer = Agent(
    llm=llm,
    role="Script Summarizer",
    goal="Compile a detailed summary of the movie {movie_name} with the new ending: {desired_ending}",
    backstory="You are tasked with writing a complete story summary for the movie {movie_name}, using the altered plot and ending: {desired_ending}. Your goal is to ensure that the summary is cohesive, well-structured, and engaging, covering all the main events and the character arcs.",
    allow_delegation=False,
    verbose=True
)

# Define tasks for agents
plan_plot = Task(
    description=(
        "1. Create a plot summary of {movie_name}, leading logically to the new ending: {desired_ending}.\n"
        "2. Focus on the core story elements, ensuring that the new ending feels like a natural progression of the story.\n"
        "3. Provide a clear plot structure with key events."
    ),
    expected_output="A comprehensive plot summary leading to the new ending.",
    agent=plot_designer
)

analyze_characters = Task(
    description=(
        "1. Analyze the main character arcs in {movie_name}.\n"
        "2. Explain how each character evolves with the new ending: {desired_ending}.\n"
        "3. Focus on the motivations, conflicts, and resolutions of key characters."
    ),
    expected_output="A summary of the character arcs, showing how they evolve with the new plot and ending.",
    agent=character_analyst
)

describe_settings = Task(
    description=(
        "1. Describe the key settings of {movie_name} and their influence on the story.\n"
        "2. Focus on how the atmosphere and locations evolve in the new ending: {desired_ending}."
    ),
    expected_output="A summary of the primary settings and their impact on the story and new ending.",
    agent=setting_designer
)

analyze_themes = Task(
    description=(
        "1. Identify the key themes in {movie_name}.\n"
        "2. Explain how the new ending: {desired_ending} reshapes or reinforces these themes.\n"
        "3. Identify any new themes introduced by the altered ending."
    ),
    expected_output="A summary of the themes and motifs in the story, focusing on the impact of the new ending.",
    agent=theme_analyst
)

summarize_script = Task(
    description=(
        "1. Compile a detailed summary of the entire movie {movie_name}, including the original plot and the new ending: {desired_ending}.\n"
        "2. Ensure the summary is cohesive and covers all major events, character arcs, and themes."
    ),
    expected_output="A detailed movie script summary with the new ending.",
    agent=script_summarizer
)

# Define the crew of agents
crew = Crew(
    agents=[plot_designer, character_analyst, setting_designer, theme_analyst, script_summarizer],
    tasks=[plan_plot, analyze_characters, describe_settings, analyze_themes, summarize_script],
    verbose=2
)

# Function to save the result as a PDF
def save_as_pdf(content, filename="Generated_Movie_Script_Summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    for line in content.split("\n"):
        pdf.multi_cell(0, 10, line)
    
    pdf.output(filename)
    return filename

# Streamlit UI for the app
st.markdown("""
                # :rainbow[TDCC Assignment 🎞️]
            """)
st.subheader("🎀 Developed by Crew AI and Meta Llama 3.1 🎀")
st.header("Movie Script Summary Generator")

with st.expander("View Crew AI Agents Details"):
    
    st.markdown('''
    <div style="font-size: 16px; line-height: 1.6;">
        <span style="color:#FF6347;"><strong>🤖 Plot Designer</strong></span><br>
        Plans an engaging and altered plot structure for the movie, seamlessly integrating the new ending while preserving key elements from the original storyline.
        <br><br>
        <span style="color:#4682B4;"><strong>🤖 Character Analyst</strong></span><br>
        Analyzes the evolution of the main characters, focusing on how they adapt and change within the context of the new storyline and conclusion.
        <br><br>
        <span style="color:#32CD32;"><strong>🤖 Setting Designer</strong></span><br>
        Describes how the settings and locations are influenced by the story’s shift, merging old and new elements to create a dynamic atmosphere.
        <br><br>
        <span style="color:#FFD700;"><strong>🤖 Theme Analyst</strong></span><br>
        Evaluates the core themes of the movie, identifying how the new ending reshapes or introduces concepts such as power, control, and the intersection of past and future.
        <br><br>
        <span style="color:#8A2BE2;"><strong>🤖 Script Summarizer</strong></span><br>
        Compiles a comprehensive story summary, ensuring all plot, character, setting, and thematic changes are cohesively presented.
    </div>
    ''', unsafe_allow_html=True)


# Inputs for movie name and alternate ending
movie_name = st.text_input("Enter the movie name", "Stree")
desired_ending = st.text_area("Enter the new ending plot twist you want", "Stree was ghost herself. She wanted to use the people of chanderi and create her own demonic empire.")

# Columns to align the buttons
col1, col2 = st.columns([3, 1])

with col1:
    generate_btn = st.button("Generate Script Summary")

# Show spinner while agents are processing
if generate_btn:
    st.session_state.processing = True
    with st.spinner("Summarizing Your Movie Script Please Wait..."):
        st.session_state.script_result = crew.kickoff(inputs={"movie_name": movie_name, "desired_ending": desired_ending})
    st.session_state.processing = False

# Display the result if available
if st.session_state.script_result:
    st.markdown(st.session_state.script_result)
    
with col2:
    if st.session_state.script_result:
        save_btn = st.button("Save as PDF")
    else:
        save_btn = None

# Save the result as PDF if the button is pressed
if save_btn and st.session_state.script_result:
    pdf_file = save_as_pdf(st.session_state.script_result)
    st.success(f"Script summary saved as {pdf_file}")
    with open(pdf_file, "rb") as file:
        st.download_button(
            label="Download PDF",
            data=file,
            file_name=pdf_file,
            mime='application/octet-stream'
        )

st.markdown(
    "<footer style='text-align: center; padding: 10px;'><strong>Made with ❤️ by Abhijit 🎀</strong></footer>", 
    unsafe_allow_html=True
)