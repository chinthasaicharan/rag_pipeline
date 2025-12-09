"""
LangChain Integration and MCP Server for Advanced RAG Pipeline
"""

from typing import List, Dict, Any, Optional
from langchain.tools import BaseTool
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document as LangChainDocument
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import BaseModel, Field
import json


# ============================================================================
# LangChain Retriever Wrapper
# ============================================================================

class AdvancedRAGRetriever(BaseRetriever):
    """LangChain-compatible retriever wrapper"""
    
    rag_retriever: Any = Field(description="RAG retriever instance")
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    use_rrf: bool = Field(default=True, description="Use Reciprocal Rank Fusion")
    use_reranking: bool = Field(default=True, description="Use reranking")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[LangChainDocument]:
        """Retrieve relevant documents"""
        results = self.rag_retriever.retrieve(
            query=query,
            top_k=self.top_k,
            use_rrf=self.use_rrf,
            use_reranking=self.use_reranking
        )
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            doc = LangChainDocument(
                page_content=result['content'],
                metadata={
                    **result['metadata'],
                    'score': result['score'],
                    'doc_id': result['doc_id']
                }
            )
            documents.append(doc)
        
        return documents


# ============================================================================
# LangChain Tools
# ============================================================================

class RAGSearchInput(BaseModel):
    """Input schema for RAG search tool"""
    query: str = Field(description="Search query to find relevant documents")
    top_k: int = Field(default=5, description="Number of results to return")


class RAGSearchTool(BaseTool):
    """LangChain tool for RAG search"""
    
    name: str = "rag_search"
    description: str = (
        "Search through ingested documents using advanced RAG techniques. "
        "Uses vector search, keyword search, Reciprocal Rank Fusion, and reranking. "
        "Useful for finding relevant information from your document collection."
    )
    args_schema: type[BaseModel] = RAGSearchInput
    rag_pipeline: Any = Field(description="RAG pipeline instance")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute RAG search"""
        results = self.rag_pipeline.search(
            query=query,
            top_k=top_k,
            use_rrf=True,
            use_reranking=True
        )
        
        if not results:
            return "No relevant documents found."
        
        # Format results
        output = []
        for i, result in enumerate(results, 1):
            output.append(
                f"Result {i} (Score: {result['score']:.4f}):\n"
                f"Source: {result['metadata'].get('file_name', 'Unknown')}\n"
                f"Content: {result['content'][:300]}...\n"
            )
        
        return "\n".join(output)


class RAGIngestInput(BaseModel):
    """Input schema for RAG ingestion tool"""
    folder_path: str = Field(description="Path to folder containing documents to ingest")


class RAGIngestTool(BaseTool):
    """LangChain tool for document ingestion"""
    
    name: str = "rag_ingest"
    description: str = (
        "Ingest documents (PDF, TXT) from a folder into the RAG database. "
        "Automatically handles deduplication. "
        "Use this to add new documents to the knowledge base."
    )
    args_schema: type[BaseModel] = RAGIngestInput
    rag_pipeline: Any = Field(description="RAG pipeline instance")
    
    class Config:
        arbitrary_types_allowed = True
    
    def _run(
        self,
        folder_path: str,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Execute document ingestion"""
        try:
            stats = self.rag_pipeline.ingest(folder_path)
            return (
                f"Ingestion complete:\n"
                f"- Processed: {stats['processed']} files\n"
                f"- Skipped (duplicates): {stats['skipped']} files\n"
                f"- Chunks added: {stats['chunks_added']}\n"
                f"- Errors: {stats['errors']}"
            )
        except Exception as e:
            return f"Error during ingestion: {str(e)}"


# ============================================================================
# MCP Server Implementation
# ============================================================================

class MCPRAGServer:
    """
    Model Context Protocol (MCP) server for RAG functionality
    Provides tools that can be called by LLMs via MCP
    """
    
    def __init__(self, rag_pipeline):
        self.rag_pipeline = rag_pipeline
        self.tools = self._register_tools()
    
    def _register_tools(self) -> Dict[str, Dict[str, Any]]:
        """Register available MCP tools"""
        return {
            "rag_search": {
                "description": "Search through document collection using advanced RAG",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results",
                            "default": 5
                        },
                        "use_rrf": {
                            "type": "boolean",
                            "description": "Use Reciprocal Rank Fusion",
                            "default": True
                        },
                        "use_reranking": {
                            "type": "boolean",
                            "description": "Use reranking",
                            "default": True
                        }
                    },
                    "required": ["query"]
                }
            },
            "rag_ingest": {
                "description": "Ingest documents from a folder",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "folder_path": {
                            "type": "string",
                            "description": "Path to document folder"
                        }
                    },
                    "required": ["folder_path"]
                }
            },
            "rag_stats": {
                "description": "Get statistics about the RAG database",
                "parameters": {
                    "type": "object",
                    "properties": {}
                }
            }
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools (MCP protocol)"""
        return [
            {
                "name": name,
                "description": tool["description"],
                "inputSchema": tool["parameters"]
            }
            for name, tool in self.tools.items()
        ]
    
    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool (MCP protocol)"""
        if tool_name == "rag_search":
            return self._handle_search(arguments)
        elif tool_name == "rag_ingest":
            return self._handle_ingest(arguments)
        elif tool_name == "rag_stats":
            return self._handle_stats()
        else:
            return {
                "error": f"Unknown tool: {tool_name}"
            }
    
    def _handle_search(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle RAG search request"""
        try:
            results = self.rag_pipeline.search(
                query=args["query"],
                top_k=args.get("top_k", 5),
                use_rrf=args.get("use_rrf", True),
                use_reranking=args.get("use_reranking", True)
            )
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_ingest(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Handle document ingestion request"""
        try:
            stats = self.rag_pipeline.ingest(args["folder_path"])
            return {
                "success": True,
                "stats": stats
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _handle_stats(self) -> Dict[str, Any]:
        """Get RAG database statistics"""
        try:
            collection = self.rag_pipeline.retriever.collection
            count = collection.count()
            
            return {
                "success": True,
                "stats": {
                    "total_chunks": count,
                    "collection_name": collection.name
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def serve_stdio(self):
        """
        Serve MCP over stdio (standard input/output)
        This allows the server to be called as a subprocess
        """
        import sys
        
        while True:
            try:
                # Read request from stdin
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line)
                
                # Handle different MCP methods
                if request.get("method") == "tools/list":
                    response = {
                        "tools": self.list_tools()
                    }
                elif request.get("method") == "tools/call":
                    params = request.get("params", {})
                    response = self.call_tool(
                        params.get("name"),
                        params.get("arguments", {})
                    )
                else:
                    response = {
                        "error": f"Unknown method: {request.get('method')}"
                    }
                
                # Send response to stdout
                sys.stdout.write(json.dumps(response) + "\n")
                sys.stdout.flush()
                
            except Exception as e:
                error_response = {"error": str(e)}
                sys.stdout.write(json.dumps(error_response) + "\n")
                sys.stdout.flush()


# ============================================================================
# Usage Examples
# ============================================================================

def example_langchain_integration():
    """Example: Using RAG with LangChain"""
    from rag_pipeline import RAGPipeline
    
    # Initialize RAG pipeline
    rag = RAGPipeline(collection_name="my_docs")
    
    # Create LangChain retriever
    retriever = AdvancedRAGRetriever(
        rag_retriever=rag.retriever,
        top_k=5,
        use_rrf=True,
        use_reranking=True
    )
    
    # Use in LangChain
    docs = retriever.get_relevant_documents("What is machine learning?")
    for doc in docs:
        print(f"Score: {doc.metadata['score']}")
        print(f"Content: {doc.page_content[:200]}...\n")
    
    # Create LangChain tools
    search_tool = RAGSearchTool(rag_pipeline=rag)
    ingest_tool = RAGIngestTool(rag_pipeline=rag)
    
    # Use tools
    result = search_tool._run("machine learning", top_k=3)
    print(result)


def example_mcp_server():
    """Example: Running MCP server"""
    from rag_pipeline import RAGPipeline
    
    # Initialize RAG pipeline
    rag = RAGPipeline(collection_name="my_docs")
    
    # Create MCP server
    server = MCPRAGServer(rag)
    
    # List available tools
    tools = server.list_tools()
    print("Available tools:", json.dumps(tools, indent=2))
    
    # Call a tool
    result = server.call_tool("rag_search", {
        "query": "machine learning",
        "top_k": 3
    })
    print("Search result:", json.dumps(result, indent=2))
    
    # Run stdio server (uncomment to actually run)
    # server.serve_stdio()


if __name__ == "__main__":
    print("=== LangChain Integration Example ===")
    example_langchain_integration()
    
    print("\n=== MCP Server Example ===")
    example_mcp_server()