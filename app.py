# app.py
import gradio as gr
import uuid
import atexit
import os
from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, RemoveMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import MessagesState, StateGraph, START, END
import sqlite3
# Import dotenv but only use if .env exists
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv()

# Base system instruction for the Arxxy persona
ARXXY_SYSTEM_INSTRUCTIONS = """
You are Arxxy, a specialized research assistant for scientific papers.  
At the start of each session, greet the user by stating your name and your purpose (end with a smiley 😊).

When the user requests research papers, follow these steps:

1. **List papers**  
   For each paper, provide:  
   - **Title** (Authors; Year)  
   - **Summary**: one-line overview  
   - **Link**: URL to the paper  
   
2. **Provide the list**

3. **Then Ask**  
   "Would you like me to include the abstracts as well?"  

4. **If yes**, re-list the papers using the same format but replace **Summary** with the full **Abstract**.
"""

# Enhanced State with context for retrieved papers
class State(MessagesState):
    summary: Optional[str] = None
    context: Optional[List[Dict[str, Any]]] = None

# Get data directory - for HF Spaces compatibility
DATA_DIR = os.environ.get("SPACE_PERSISTENT_DIR", "/mnt/data")
PERSIST_DIR = os.path.join(DATA_DIR, "chroma_db")

# Initialize SQLite for conversation persistence
DB_PATH = os.path.join(DATA_DIR, "conversation.db")
conn = sqlite3.connect(DB_PATH, check_same_thread=False)

# Register connection close on program exit
@atexit.register
def close_connection():
    print("Closing SQLite connection...")
    conn.close()

# Initialize shared components
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0)
vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# Node functions
def call_model(state: State):
    summary = state.get("summary", "")
    
    # Start with the base Arxxy persona instructions
    system_content = ARXXY_SYSTEM_INSTRUCTIONS
    
    # Add summary if available
    if summary:
        system_content += f"\n\nSummary of earlier conversation: {summary}"
    
    # Create messages with system instructions
    messages = [SystemMessage(content=system_content)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": response}

def summarizer(state: State):
    summary = state.get("summary", "")
    
    if summary:
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = "Create a summary of the conversation above:"
    
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = llm.invoke(messages)
    
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

def paper_retriever(state: State):
    messages = state["messages"]
    last_message = messages[-1]
    query = last_message.content
    
    docs = retriever.invoke(query)
    
    paper_context = []
    for doc in docs:
        paper_info = {
            "id": doc.metadata.get("id", "Unknown"),
            "title": doc.metadata.get("title", "Unknown Title"),
            "authors": doc.metadata.get("authors", "Unknown Authors"),
            "categories": doc.metadata.get("categories", ""),
            "content": doc.page_content
        }
        paper_context.append(paper_info)
    
    return {"context": paper_context}

def enhanced_response_node(state: State):
    messages = state["messages"]
    summary = state.get("summary", "")
    paper_context = state.get("context", [])
    
    # Start with the base Arxxy persona instructions
    system_content = ARXXY_SYSTEM_INSTRUCTIONS
    
    # Add summary if available
    if summary:
        system_content += f"\n\nSummary of earlier conversation: {summary}"
    
    # Add paper context if available
    if paper_context:
        system_content += "\n\nRelevant papers:"
        for i, paper in enumerate(paper_context):
            paper_info = f"""
            \n--- Paper {i+1} ---
            ID: {paper['id']}
            Title: {paper['title']}
            Authors: {paper['authors']}
            Categories: {paper['categories']}
            Content: {paper['content']}
            """
            system_content += paper_info
    
    enhanced_messages = [SystemMessage(content=system_content)] + messages
    
    response = llm.invoke(enhanced_messages)
    
    return {"messages": response}

def router(state: State):
    messages = state["messages"]
    
    if len(messages) > 10:
        return "summarizer"
    
    last_message = messages[-1]
    
    if hasattr(last_message, "content") and isinstance(last_message.content, str):
        content = last_message.content.lower()
        research_keywords = [
            "paper", "research", "publication", "article", "arxiv", 
            "find", "search", "study", "physics", "quantum"
        ]
        if any(keyword in content for keyword in research_keywords):
            return "paper_retriever"
    
    return "END"

# Create the graph - initialize at module level
def build_conversation_graph():
    builder = StateGraph(State)
    
    # Add nodes
    builder.add_node("call_model", call_model)
    builder.add_node("summarizer", summarizer)
    builder.add_node("paper_retriever", paper_retriever)
    builder.add_node("enhanced_response_node", enhanced_response_node)
    
    # Set up the flow
    builder.add_edge(START, "call_model")
    
    # Add conditional edge after model response
    builder.add_conditional_edges(
        "call_model",
        router,
        {
            "summarizer": "summarizer",
            "paper_retriever": "paper_retriever",
            "END": END
        }
    )
    
    # Connect paper retrieval to enhanced response
    builder.add_edge("paper_retriever", "enhanced_response_node")
    builder.add_edge("enhanced_response_node", END)
    
    # Complete the summarization path - loop back to call_model after summarizing
    # This allows the model to respond after summarization instead of ending the conversation
    builder.add_edge("summarizer", "call_model")
    
    # Create memory checkpoint and compile
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory)

# Initialize graph at module level
conversation_graph = build_conversation_graph()

# Gradio chat interface handler
def respond(message, history, thread_id):
    # Create checkpoint config - simplified
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "gradio-chat"
        }
    }
    
    # Create input with ONLY the new message - this is crucial
    # Let LangGraph's checkpointer handle adding to existing messages
    input_state = {"messages": [HumanMessage(content=message)]}
    
    # Use the pre-compiled graph
    # LangGraph will:
    # 1. Load the existing state from checkpointer (with previous messages, summary, context)
    # 2. Add the new message to the existing messages via add_messages
    # 3. Process the updated state through the graph
    result = conversation_graph.invoke(input_state, config)
    
    # Get AI response - our graph is designed to always end with an AI message
    last_message = result["messages"][-1]
    
    if isinstance(last_message, AIMessage):
        # Return in format expected by Gradio messages type
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": last_message.content}
        ]
    else:
        # Fallback in case the last message isn't an AIMessage
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "I couldn't generate a response. Please try again."}
        ]

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="ArXiv Research Assistant") as demo:
        gr.Markdown("# ArXiv Research Assistant")
        gr.Markdown("Ask me anything about scientific papers!")
        
        # Generate a unique session ID for each user
        session_id = gr.State(value=str(uuid.uuid4()))
        
        chatbot = gr.Chatbot(height=500, type="messages")
        msg = gr.Textbox(placeholder="Type your message here...", show_label=False)
        clear = gr.Button("Clear")
        
        # Pass session_id to the respond function
        msg.submit(respond, [msg, chatbot, session_id], [chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)
    
    return demo

# Main entry point
if __name__ == "__main__":
    demo = create_interface()
    # HF Spaces compatible launch with share=False
    demo.launch(server_name="0.0.0.0", share=False)