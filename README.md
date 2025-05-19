# Research Assistant

A conversational AI assistant for searching and discussing research papers.

## Features
- Search for relevant papers based on user queries
- Provide detailed information about research papers
- Token management through context summarization
- Routing between general conversation and research assistance

## Technologies
- LangChain framework for LLM interactions
- LangGraph for conversation flow management
- SQLite for conversation persistence
- ChromaDB for vector storage and retrieval
- Gradio for the web interface

## App Architecture
The application uses a state machine graph structure with these key components:
- `call_model`: Handles basic conversation
- `router`: Directs flow based on message content
- `paper_retriever`: Fetches relevant research papers
- `enhanced_response_node`: Provides detailed paper responses
- `summarizer`: Creates conversation summaries when needed
  
```mermaid
flowchart TD
    A["START"] --> B("call_model")
    B -- Router Logic Evaluates State --> C{"router"}
    C -- len(messages) > 10 --> D("summarizer")
    D -- Loops back after summarizing --> B
    C -- Keyword match in last message --> E("paper_retriever")
    E -- After paper retrieval --> F("enhanced_response_node")
    F -- After enhanced response --> G["END"]
    C -- Default: No other conditions met --> G

     A:::specialNodeStyle
     B:::processNodeStyle
     C:::routerStyle
     D:::processNodeStyle
     E:::processNodeStyle
     F:::processNodeStyle
     G:::specialNodeStyle
    classDef nodeStyle fill:#f9f,stroke:#333,stroke-width:2px
    classDef processNodeStyle fill:#lightgrey,stroke:#333,stroke-width:2px
    classDef specialNodeStyle fill:#orange,stroke:#333,stroke-width:2px
    classDef routerStyle fill:#lightblue,stroke:#333,stroke-width:2px

```



