"""Sample implementation to validate the LangGraph chat setup."""
import os
from dotenv import load_dotenv
from chat import create_chat_graph
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

# Verify API key is set
if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY not set. Please set it in .env file")

def test_basic_chat():
    """Test basic chat functionality."""
    print("=" * 60)
    print("Testing LangGraph Chat with Claude Haiku 3.5")
    print("=" * 60)
    
graph = create_chat_graph()
    
    # Test conversation
    test_messages = [
        "Hello! What is LangGraph?",
        "Can you explain state graphs in simple terms?",
        "Give me a code example of a simple state graph",
    ]
    
    messages = []
    
    for i, user_input in enumerate(test_messages, 1):
        print(f"\n--- Test {i} ---")
        print(f"User: {user_input}")
        
        # Add user message
        messages.append(HumanMessage(content=user_input))
        
        # Run the graph
        state = {"messages": messages}
        result = graph.invoke(state)
        
        # Extract and display assistant response
        assistant_message = result["messages"][-1]
        print(f"Assistant: {assistant_message.content}")
        
        # Update messages for next iteration
        messages = result["messages"]
    
    print("\n" + "=" * 60)
    print("✅ Chat validation completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_chat()