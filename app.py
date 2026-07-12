import streamlit as st
import os
from typing import TypedDict, List
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. STREAMLIT INITIAL INITIALIZATION & SECRETS
# ==========================================
st.set_page_config(page_title="Technical Support AI Assistant", layout="wide")
st.title("🛠️ Technical Support AI Agent")
st.caption("Powered by LangGraph, Qdrant Cloud, and Groq (Llama 3.1)")

# Ensure cloud secrets are available (For local testing, you can use os.environ)
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    QDRANT_URL = st.secrets["QDRANT_URL"]
    QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
except Exception:
    st.error("Missing cloud configuration secrets! Please configure them in Streamlit.")
    st.stop()

# ==========================================
# 2. STATE & GRAPH DEFINITIONS
# ==========================================
class AgentState(TypedDict):
    messages: List[BaseMessage]
    needs_human: bool
    current_query: str
    solution: str

# Triage Node: Screens queries for engineering frustration or explicit handoff keywords
def triage_node(state: AgentState):
    query = state["current_query"].lower()
    human_keywords = ["human", "agent", "representative", "speak to someone", "person", "escalate"]
    
    if any(keyword in query for keyword in human_keywords):
        return {"needs_human": True}
    return {"needs_human": False}

# RAG Engine Node: Pulls documentation from cloud Qdrant and synthesizes an answer using Groq
def rag_bot_node(state: AgentState):
    # Initialize Core Engines inside the node execution state
    llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=GROQ_API_KEY)
    
    dense_embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    vector_store = QdrantVectorStore(
        client=client,
        collection_name="tech_docs",
        embedding=dense_embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=RetrievalMode.HYBRID
    )
    
    # Retrieve documents matched by Hybrid Querying
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    matched_docs = retriever.invoke(state["current_query"])
    
    context = "\n\n".join([doc.page_content for doc in matched_docs])
    
    system_prompt = (
        "You are an expert technical support engineer. Provide deep troubleshooting steps using the context. "
        "If the context doesn't contain the answer, say you don't know.\n\nContext:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": context, "input": state["current_query"]})
    
    return {
        "messages": state["messages"] + [AIMessage(content=response.content)],
        "solution": response.content
    }

# Human Escape Hatch Node
def human_queue_node(state: AgentState):
    escalation_msg = (
        "🚨 **SYSTEM ESCALATION:** I am opening a ticket and routing you directly to a human support engineer. "
        "An agent will review your log context shortly."
    )
    return {"messages": state["messages"] + [AIMessage(content=escalation_msg)]}

# Dynamic Router Conditional Edge
def routing_edge(state: AgentState):
    return "human_queue" if state["needs_human"] else "rag_bot"

# Assemble the LangGraph
@st.cache_resource
def compile_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("triage", triage_node)
    workflow.add_node("rag_bot", rag_bot_node)
    workflow.add_node("human_queue", human_queue_node)
    
    workflow.set_entry_point("triage")
    workflow.add_conditional_edges("triage", routing_edge, {"human_queue": "human_queue", "rag_bot": "rag_bot"})
    workflow.add_edge("rag_bot", END)
    workflow.add_edge("human_queue", END)
    
    return workflow.compile(checkpointer=MemorySaver())

graph_app = compile_graph()

# ==========================================
# 3. CHAT INTERFACE MANAGEMENT
# ==========================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display previous messaging elements
for message in st.session_state.chat_history:
    with st.chat_message("user" if isinstance(message, HumanMessage) else "assistant"):
        st.markdown(message.content)

# Handle incoming user query
if user_input := st.chat_input("Type your technical error code or question..."):
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    
    # Configure thread tracking dynamically
    config = {"configurable": {"thread_id": "demo_session_1"}}
    
    initial_input = {
        "messages": st.session_state.chat_history,
        "current_query": user_input,
        "needs_human": False,
        "solution": ""
    }
    
    # Run the graph synchronously for UI presentation
    output_state = graph_app.invoke(initial_input, config=config)
    
    final_message = output_state["messages"][-1].content
    
    with st.chat_message("assistant"):
        st.markdown(final_message)
        
    st.session_state.chat_history.append(AIMessage(content=final_message))