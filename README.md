# Zero Infrastructure RAG Agent

This Python script demonstrates a zero-infrastructure RAG agent. It uses local Ollama for LLM inference and HuggingFace for embeddings, processing a mock knowledge base to answer questions without requiring cloud services or complex setup.

## Language

`python`

## How to Run

1. Install dependencies: pip install langchain-community langchain-huggingface sentence-transformers
2. Download an Ollama model: ollama pull llama3
3. Run the script: python zero_infra_rag.py

## Original Article

This example accompanies the Turkish article: [Sıfır Altyapılı RAG Ajanları: Bilgi Tabanları ve Çok Modlu Bağlam İşleme (MCP) ile Güçlendirme](https://fatihsoysal.com/blog/sifir-altyapili-rag-ajanlari-bilgi-tabanlari-ve-cok-modlu-baglam-isleme-mcp-ile-guclendirme/).

## License

MIT — see [LICENSE](LICENSE).
