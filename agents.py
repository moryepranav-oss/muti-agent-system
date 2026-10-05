from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# ---------- Model setup ----------
# Model for the agents that call tools (good at tool calling, has its own token limit)
agent_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Model for the writer and critic (no tools, so token use is controlled)
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    max_tokens=1500,
    reasoning_effort="low",
)


# Separate small model for the critic: it has its own token budget,
# so it does not compete with the writer inside the same minute.
critic_llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, max_tokens=700)


# 1st agent
def build_search_agent():
    return create_agent(
        model=agent_llm,
        tools=[web_search],
    )


# 2nd agent
def build_reader_agent():
    return create_agent(
        model=agent_llm,
        tools=[scrape_url],
    )


# writer chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# critic chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | critic_llm | StrOutputParser()