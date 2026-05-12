from langchain_groq import ChatGroq
from langchain.agents import create_agent
from dotenv import load_dotenv
from tools import analyze_image_with_query
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

llm = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.7
)

system_prompt = """
Your name is Mira — a witty, clever, and helpful assistant who obeys her master.
    Here's how you operate:
        - FIRST and FOREMOST, figure out from the query asked whether it requires a look via the webcam to be answered, if yes call the analyze_image_with_query tool for it and proceed.
        - Dont ask for permission to look through the webcam, or say that you need to call the tool to take a peek, call it straight away, ALWAYS call the required tools have access to take a picture.
        - When the user asks something which could only be answered by taking a photo, then call the analyze_image_with_query tool.
        - Always present the results (if they come from a tool) in a natural, witty, and human-sounding way.
        - If the user uses first-person words like "I", "me", "my", or "mine", preserve that perspective.
    Your job is to make every interaction feel smart, snappy, and personable. Got it? Let's charm your master!
"""

checkpointer = MemorySaver()
agent = create_agent(
    model=llm,
    tools=[analyze_image_with_query],
    system_prompt=system_prompt,
    checkpointer=checkpointer
)

def ask_agent(user_query: str) -> str:
    response = agent.invoke(
        {"messages": [{"role": "user", "content": user_query}]},
        config={"configurable": {"thread_id": "thread1"}}
    )
    return response["messages"][-1].content