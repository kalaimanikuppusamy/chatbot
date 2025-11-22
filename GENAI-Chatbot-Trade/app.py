import os

from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing_extensions import TypedDict
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# vector store (Chroma)
from langchain.vectorstores import Chroma

from langchain.tools import tool
import logging

logging.basicConfig( level=logging.INFO )
logger = logging.getLogger( __name__ )

PERSIST_DIR = "chroma_db"


def load_and_split_pdfs(folder_path):
    """Load PDFs from folder and split into chunks."""
    if not os.path.exists( folder_path ):
        logger.warning( f"Folder {folder_path} does not exist. Creating it..." )
        os.makedirs( folder_path, exist_ok=True )
        return []

    docs = []
    pdf_files = [f for f in os.listdir( folder_path ) if f.endswith( ".pdf" )]

    if not pdf_files:
        logger.warning( f"No PDF files found in {folder_path}" )
        return []

    logger.info( f"Found {len( pdf_files )} PDF files" )

    for file in pdf_files:
        try:
            file_path = os.path.join( folder_path, file )
            logger.info( f"Loading {file}..." )
            loader = PyPDFLoader( file_path )
            file_docs = loader.load()
            docs.extend( file_docs )
            logger.info( f"Loaded {len( file_docs )} pages from {file}" )
        except Exception as e:
            logger.error( f"Error loading {file}: {str( e )}" )

    if not docs:
        logger.warning( "No documents were loaded successfully" )
        return []

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    split_docs = splitter.split_documents( docs )
    return split_docs


def create_or_load_vectorstore(documents=None, name="docs"):
    model_name = "BAAI/bge-small-en-v1.5"  # "BAAI/bge-base-en-v1.5"
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},  # or "cuda" for GPU
        encode_kwargs={"normalize_embeddings": True}
    )

    # # If DB exists, load it
    # if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
    #     print("✅ Found existing ChromaDB. Loading without re-embedding...")
    #     return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)

    # Otherwise, create it
    print( "📄 Creating ChromaDB and embedding documents..." )
    vectordb = Chroma.from_documents( documents, embeddings, persist_directory=PERSIST_DIR, collection_name=name )
    vectordb.persist()
    return vectordb


invoice_docs = load_and_split_pdfs( "invoices" )
order_docs = load_and_split_pdfs( "orders" )
invoice_vectordb = create_or_load_vectorstore( invoice_docs, "invoices_pdfs" )
order_vectordb = create_or_load_vectorstore( order_docs, "orders_pdfs" )

orders_retriever = order_vectordb.as_retriever( search_type="similarity", search_kwargs={"k": 4} )
invoices_retriever = invoice_vectordb.as_retriever( search_type="similarity", search_kwargs={"k": 4} )


# RAG Tool for orders PDFs
@tool
def search_orders(query: str) -> str:
    """Search the orders PDFs for information on order details, shipping details, customer details, shipper details, products and number of orders """
    print( f"Inside search_orders {query}" )
    docs = orders_retriever.get_relevant_documents( query )
    print( docs )
    if not docs:
        return "No matches found."
    chunks = []
    for i, d in enumerate( docs, 1 ):
        meta = d.metadata or {}
        src = meta.get( "source", "unknown" )
        page = meta.get( "page", None )
        where = f" (page {page})" if page is not None else ""
        chunks.append( f"[{i}] {src}{where}:\n{d.page_content[:800]}" )
    return "\n\n".join( chunks )


# RAG tool for invoices PDFs
@tool
def search_invoices(query: str) -> str:
    """Search the invoices PDFs for information on invoice details"""
    docs = invoices_retriever.get_relevant_documents( query )
    if not docs:
        return "No matches found."
    chunks = []
    for i, d in enumerate( docs, 1 ):
        meta = d.metadata or {}
        src = meta.get( "source", "unknown" )
        page = meta.get( "page", None )
        where = f" (page {page})" if page is not None else ""
        chunks.append( f"[{i}] {src}{where}:\n{d.page_content[:800]}" )
    return "\n\n".join( chunks )


# Fictional API call to get the order status
@tool( "get_order_status", return_direct=False, description="Get the order status of an order id" )
def get_order_status(order_id: str) -> str:
    statuses = ["PENDING", "PROCESSING", "SHIPPED", "DELIVERED", "CANCELLED"]
    idx = sum( ord( c ) for c in order_id ) % len( statuses )
    return f"Order {order_id} status: {statuses[idx]}"


class QueryType( BaseModel ):
    query_type: Literal["invoice", "order", "general"] = Field(
        ...,
        description="Classify if the message was a query about invoices, orders or a general query. If the query is about invoices return invoice, else If the query is about orders return order, else return general"
    )


class State( TypedDict ):
    messages: Annotated[list, add_messages]
    query_type: str | None


llm = ChatOllama( model="llama3.2", temperature=0.7 )

orders_tools = [search_orders, get_order_status]
invoices_tools = [search_invoices]


def chatbot(state: State):
    return {"messages": [llm.invoke( state["messages"] )]}


def classify_query(state: State):
    message = state["messages"][-1]

    classifier_llm = llm.with_structured_output( QueryType )
    result = classifier_llm.invoke( [
        {
            "role": "system",
            "content": """Classify the user message as either questions on invoices, orders or general.
            - 'order': If the user asks about an order or the query has the word order or orders in it. Or the number of orders or mentions order date or order details
            - 'invoice': If the user asks about an invoice or the query has the word invoice or invoices or mentions invoice date or invoice details
            - 'general' : If the user asks anything other than about invoices or orders """
        },
        {
            "role": "user",
            "content": message.content
        }
    ] )
    print( result.query_type )
    return {"query_type": result.query_type}


def router(state: State):
    query_type = state.get( "query_type", "general" )
    print( f"ROUTER {query_type}" )
    if query_type == "order":
        return {"next": "order"}
    elif query_type == "invoice":
        return {"next": "invoice"}
    return {"next": "general"}


def orders_agent(state: State):
    """Handle tool calls and responses properly"""
    llm_with_tools = llm.bind_tools( orders_tools )

    # Build proper message history
    messages = [
                   {"role": "system",
                    "content": "You are an assistant that helps with order queries. Use the available tools when needed to search for order information or check order status."}
               ] + state["messages"]

    response = llm_with_tools.invoke( messages )
    print( f"Orders agent response: {response}" )

    # Check if the model wants to use tools
    if response.tool_calls:
        # Process tool calls
        tool_messages = []
        for tool_call in response.tool_calls:
            print( f"Tool call: {tool_call}" )

            # Find and execute the tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "search_orders":
                result = search_orders( tool_args.get( 'query' ) )
            elif tool_name == "get_order_status":
                result = get_order_status( tool_args.get( 'order_id' ) )
            else:
                result = f"Unknown tool: {tool_name}"

            # Create tool message
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append( tool_message )

        # Get final response with tool results
        final_messages = messages + [response] + tool_messages
        final_response = llm.invoke( final_messages )

        return {"messages": [response] + tool_messages + [final_response]}
    else:
        # No tools needed
        return {"messages": [response]}


def general_agent(state: State):
    last_message = state["messages"][-1]

    messages = [
        {"role": "system",
         "content": """You are a business internal employees assistant focusing on helping them with orders, invoices or other
          common queries they may have."""
         },
        {
            "role": "user",
            "content": last_message.content
        }
    ]
    response = llm.invoke( messages )

    return {"messages": [response]}


def invoices_agent(state: State):
    """Handle tool calls and responses properly for invoices"""
    llm_with_tools = llm.bind_tools( invoices_tools )

    # Build proper message history
    messages = [
                   {"role": "system",
                    "content": "You are an assistant that helps with invoice queries. Use the available tools when needed to search for invoice information."}
               ] + state["messages"]

    response = llm_with_tools.invoke( messages )
    print( f"Invoice agent response: {response}" )

    # Check if the model wants to use tools
    if response.tool_calls:
        # Process tool calls
        tool_messages = []
        for tool_call in response.tool_calls:
            print( f"Tool call: {tool_call}" )

            # Find and execute the tool
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name == "search_invoices":

                result = search_invoices( tool_args.get( 'query' ) )
            else:
                result = f"Unknown tool: {tool_name}"

            # Create tool message
            tool_message = ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
            tool_messages.append( tool_message )

        # Get final response with tool results
        final_messages = messages + [response] + tool_messages
        final_response = llm.invoke( final_messages )

        return {"messages": [response] + tool_messages + [final_response]}
    else:
        # No tools needed
        return {"messages": [response]}


graph_builder = StateGraph( State )
graph_builder.add_node( "classifier", classify_query )
graph_builder.add_node( "router", router )
graph_builder.add_node( "order", orders_agent )
graph_builder.add_node( "invoice", invoices_agent )
graph_builder.add_node( "general", general_agent )

graph_builder.add_edge( START, "classifier" )
graph_builder.add_edge( "classifier", "router" )
graph_builder.add_conditional_edges( "router",
                                     lambda state: state.get( "next" ),
                                     {"order": "order", "invoice": "invoice", "general": "general"}
                                     )
graph_builder.add_edge( "order", END )
graph_builder.add_edge( "invoice", END )
graph_builder.add_edge( "general", END )

graph = graph_builder.compile()


def run():
    state = {"messages": [], "query_type": None}

    while True:
        user_input = input( "Message: " )
        if user_input == "exit":
            print( "Bye.." )
            break

        # Create proper message object
        user_message = HumanMessage( content=user_input )
        state["messages"] = state.get( "messages", [] ) + [user_message]

        # Invoke the graph
        result = graph.invoke( state )

        # Update state with results
        state = result

        # Print the last assistant message
        if result.get( "messages" ) and len( result["messages"] ) > 0:
            last_message = result["messages"][-1]
            if hasattr( last_message, 'content' ):
                print( f"Assistant: {last_message.content}" )
            else:
                print( f"Assistant: {last_message}" )


if __name__ == "__main__":
    run()
