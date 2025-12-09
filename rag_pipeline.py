"""
Advanced RAG Pipeline with Reciprocal Rank Fusion and Reranking
Modular design for easy integration with LangChain and MCP servers
"""

import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import numpy as np


@dataclass
class Document:
    """Document structure for internal use"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str


class DocumentIngestion:
    """Handles document ingestion with deduplication"""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
    def _compute_file_hash(self, file_path: str) -> str:
        """Compute MD5 hash of file for deduplication"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def _is_file_ingested(self, file_hash: str) -> bool:
        """Check if file already exists in database"""
        results = self.collection.get(
            where={"file_hash": file_hash},
            limit=1
        )
        return len(results['ids']) > 0
    
    def _load_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Load document based on file type"""
        file_ext = Path(file_path).suffix.lower()
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif file_ext == '.txt':
                loader = TextLoader(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_ext}")
            
            docs = loader.load()
            return docs
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return []
    
    def ingest_folder(
        self,
        folder_path: str,
        file_extensions: List[str] = ['.pdf', '.txt']
    ) -> Dict[str, int]:
        """
        Ingest all documents from a folder with deduplication
        
        Args:
            folder_path: Path to folder containing documents
            file_extensions: List of file extensions to process
            
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            'processed': 0,
            'skipped': 0,
            'chunks_added': 0,
            'errors': 0
        }
        
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")
        
        for file_path in folder.rglob('*'):
            if file_path.suffix.lower() not in file_extensions:
                continue
            
            try:
                file_hash = self._compute_file_hash(str(file_path))
                
                # Check if already ingested
                if self._is_file_ingested(file_hash):
                    print(f"Skipping already ingested: {file_path.name}")
                    stats['skipped'] += 1
                    continue
                
                # Load and process document
                docs = self._load_document(str(file_path))
                if not docs:
                    stats['errors'] += 1
                    continue
                
                # Split into chunks
                chunks = self.text_splitter.split_documents(docs)
                
                # Prepare for Chroma
                texts = [chunk.page_content for chunk in chunks]
                metadatas = []
                ids = []
                
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{file_hash}_{i}"
                    metadata = {
                        'source': str(file_path),
                        'file_name': file_path.name,
                        'file_hash': file_hash,
                        'chunk_index': i,
                        'total_chunks': len(chunks),
                        **chunk.metadata
                    }
                    metadatas.append(metadata)
                    ids.append(chunk_id)
                
                # Add to collection
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                stats['processed'] += 1
                stats['chunks_added'] += len(chunks)
                print(f"Ingested: {file_path.name} ({len(chunks)} chunks)")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                stats['errors'] += 1
        
        return stats
    
    def clear_collection(self):
        """Delete all documents from collection"""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )


class AdvancedRetriever:
    """Advanced retriever with Reciprocal Rank Fusion and reranking"""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
    
    def _reciprocal_rank_fusion(
        self,
        results_list: List[List[Tuple[str, float, Dict]]],
        k: int = 60
    ) -> List[Tuple[str, float, Dict]]:
        """
        Combine multiple ranked lists using Reciprocal Rank Fusion
        
        Args:
            results_list: List of ranked results [(doc_id, score, metadata), ...]
            k: Constant for RRF formula (default 60)
            
        Returns:
            Fused and re-ranked results
        """
        doc_scores = {}
        doc_metadata = {}
        
        for results in results_list:
            for rank, (doc_id, score, metadata) in enumerate(results, start=1):
                if doc_id not in doc_scores:
                    doc_scores[doc_id] = 0
                    doc_metadata[doc_id] = metadata
                # RRF formula: 1 / (k + rank)
                doc_scores[doc_id] += 1 / (k + rank)
        
        # Sort by fused score
        fused_results = [
            (doc_id, score, doc_metadata[doc_id])
            for doc_id, score in sorted(
                doc_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        
        return fused_results
    
    def _vector_search(self, query: str, n_results: int = 10) -> List[Tuple[str, float, Dict]]:
        """Perform vector similarity search"""
        query_embedding = self.embeddings.embed_query(query)
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        formatted_results = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                doc_id = results['ids'][0][i]
                distance = results['distances'][0][i]
                metadata = results['metadatas'][0][i]
                # Convert distance to similarity score
                score = 1 / (1 + distance)
                formatted_results.append((doc_id, score, metadata))
        
        return formatted_results
    
    def _keyword_search(self, query: str, n_results: int = 10) -> List[Tuple[str, float, Dict]]:
        """Perform keyword-based search (BM25-like)"""
        # Get all documents
        all_docs = self.collection.get(include=['documents', 'metadatas'])
        
        if not all_docs['ids']:
            return []
        
        # Simple keyword scoring (can be enhanced with BM25)
        query_terms = set(query.lower().split())
        scored_docs = []
        
        for i, doc in enumerate(all_docs['documents']):
            doc_terms = set(doc.lower().split())
            # Calculate overlap score
            overlap = len(query_terms & doc_terms)
            if overlap > 0:
                score = overlap / len(query_terms)
                scored_docs.append((
                    all_docs['ids'][i],
                    score,
                    all_docs['metadatas'][i]
                ))
        
        # Sort and return top results
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:n_results]
    
    def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[Tuple[str, float, Dict]],
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Rerank results using cross-encoder (lightweight version)
        For production, consider using sentence-transformers cross-encoders
        """
        # Get document contents
        doc_ids = [r[0] for r in results]
        docs = self.collection.get(ids=doc_ids, include=['documents'])
        
        # Simple relevance scoring based on query term frequency
        reranked = []
        query_terms = query.lower().split()
        
        for i, (doc_id, orig_score, metadata) in enumerate(results):
            doc_content = docs['documents'][i].lower()
            
            # Count query term occurrences
            relevance = sum(doc_content.count(term) for term in query_terms)
            # Combine with original score
            combined_score = orig_score * 0.5 + (relevance / max(len(doc_content.split()), 1)) * 0.5
            
            reranked.append((doc_id, combined_score, metadata))
        
        # Sort by combined score
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked[:top_k]
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        use_rrf: bool = True,
        use_reranking: bool = True,
        n_results_per_method: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using advanced techniques
        
        Args:
            query: Search query
            top_k: Number of final results to return
            use_rrf: Whether to use Reciprocal Rank Fusion
            use_reranking: Whether to apply reranking
            n_results_per_method: Number of results to get from each method
            
        Returns:
            List of retrieved documents with metadata
        """
        if use_rrf:
            # Get results from multiple methods
            vector_results = self._vector_search(query, n_results_per_method)
            keyword_results = self._keyword_search(query, n_results_per_method)
            
            # Fuse results
            results = self._reciprocal_rank_fusion(
                [vector_results, keyword_results]
            )
        else:
            # Use only vector search
            results = self._vector_search(query, n_results_per_method)
        
        # Apply reranking if requested
        if use_reranking and results:
            results = self._rerank_with_cross_encoder(query, results, top_k)
        else:
            results = results[:top_k]
        
        # Get full document content
        if not results:
            return []
        
        doc_ids = [r[0] for r in results]
        docs = self.collection.get(ids=doc_ids, include=['documents', 'metadatas'])
        
        # Format final results
        final_results = []
        for i, (doc_id, score, _) in enumerate(results):
            final_results.append({
                'content': docs['documents'][i],
                'metadata': docs['metadatas'][i],
                'score': score,
                'doc_id': doc_id
            })
        
        return final_results


class RAGPipeline:
    """Complete RAG pipeline for easy integration"""
    
    def __init__(
        self,
        collection_name: str = "rag_documents",
        persist_directory: str = "./chroma_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.ingestion = DocumentIngestion(
            collection_name=collection_name,
            persist_directory=persist_directory,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.retriever = AdvancedRetriever(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_model=embedding_model
        )
    
    def ingest(self, folder_path: str) -> Dict[str, int]:
        """Ingest documents from folder"""
        return self.ingestion.ingest_folder(folder_path)
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_rrf: bool = True,
        use_reranking: bool = True
    ) -> List[Dict[str, Any]]:
        """Search for relevant documents"""
        return self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_rrf=use_rrf,
            use_reranking=use_reranking
        )
    
    def get_retriever_for_langchain(self):
        """Get retriever instance for LangChain integration"""
        return self.retriever


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    rag = RAGPipeline(
        collection_name="my_documents",
        persist_directory="./my_chroma_db"
    )
    
    # Ingest documents (only new files will be added)
    stats = rag.ingest("./documents")
    print(f"Ingestion complete: {stats}")
    
    # Search with advanced retrieval
    results = rag.search(
        query="SSC",
        top_k=3,
        use_rrf=True,
        use_reranking=True
    )
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {result['score']:.4f}) ---")
        print(f"Source: {result['metadata']['file_name']}")
        print(f"Content: {result['content'][:200]}...")