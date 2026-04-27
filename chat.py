from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import TypedDict


class ChatState(TypedDict):
    """State for the chat graph."""
    messages: Annotated[list[BaseMessage], "The conversation messages"]


def create_chat_graph():
    """Create a LangGraph chat application with Claude Haiku."""
    
    # Initialize Claude Haiku model
    model = ChatAnthropic(model="claude-3-5-haiku-20241022")
    
    # Create the state graph
    graph_builder = StateGraph(ChatState)
    
    def chat_node(state: ChatState) -> ChatState:
        """Process messages and generate responses."""
        response = model.invoke(state["messages"])
        return {
            "messages": state["messages"] + [response]
        }
    
    # Add nodes
    graph_builder.add_node("chat", chat_node)
    
    # Add edges
    graph_builder.add_edge(START, "chat")
    graph_builder.add_edge("chat", END)
    
    # Compile the graph
    graph = graph_builder.compile()
    return graph


def run_chat():
    """Run an interactive chat session."""
    graph = create_chat_graph()
    messages = []
    
    print("🤖 LangGraph Chat with Claude Haiku")
    print("Type 'exit' to quit\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Run the graph
        state = {"messages": messages}
        result = graph.invoke(state)
        
        # Extract and display assistant response
        assistant_message = result["messages"][-1]
        print(f"Assistant: {assistant_message.content}\n")
        
        # Update messages for next iteration
        messages = result["messages"]


if __name__ == "__main__":
    run_chat()