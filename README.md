# Categorizador Backend

This repository contains a Django based API that manages document uploads and metadata analysis. The
project stores vectors in [Weaviate](https://weaviate.io/) and metadata in
[Neo4j](https://neo4j.com/). It also integrates with local LLMs via
[Ollama](https://github.com/jmorganca/ollama) for embedding and text generation tasks.

## Objectives

* Centralize different types of content (text, images, audio and video) in a single knowledge graph.
* Extract embeddings for advanced search and retrieval across multiple modalities.
* Provide endpoints to process documents, connect nodes and explore relationships.
* Facilitate future experimentation with voice/music/video embeddings and graph algorithms.

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd categorizador-backend
   ```
2. **Create a virtual environment** (Python 3.10+)
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. **Install dependencies** using `poetry` or `pip`:
   ```bash
   pip install -U pip
   pip install -e .
   # or with poetry
   # poetry install
   ```
4. **Apply migrations** and seed initial data:
   ```bash
   python manage.py migrate
   python manage.py seed_initial_data
   ```
5. **Run the development server**
   ```bash
   python manage.py runserver
   ```

Adjust environment variables such as `NEO4J_URI`, `WEAVIATE_HTTP_HOST` and `LLM_BASE_URL` if needed.

## API Usage

The API is served under `/api/` and provides endpoints for uploading files,
processing metadata, searching the graph and interacting with the LLM. Some
notable routes include:

| Method | Endpoint                           | Description                                 |
| ------ | ---------------------------------- | ------------------------------------------- |
| POST   | `/files/upload`                    | Upload one or more files.                   |
| GET    | `/files/`                          | List pending uploaded files.                |
| POST   | `/metadata/save`                   | Process file metadata and create embeddings |
| POST   | `/nodes/create-node`               | Create a node in Neo4j.                     |
| POST   | `/graph`                           | Retrieve a subgraph of connected nodes.     |
| POST   | `/graph/search/`                   | Advanced search using custom Cypher.        |
| POST   | `/content/process`                 | Process raw text for embeddings.            |
| POST   | `/search`                          | Multi‑modal search across text and images.  |

For full details check `api/urls.py` and the corresponding views in `api/`.

## Management Commands

Two custom Django commands help bootstrap and reset the environment:

* `seed_initial_data` – creates upload folders, seeds the base node types in Neo4j and the Weaviate schema.
  ```bash
  python manage.py seed_initial_data
  ```
* `reset_app_data` – clears Neo4j, SQLite and Weaviate content and deletes uploaded files (requires `--confirm`).
  ```bash
  python manage.py reset_app_data --confirm
  ```

## Upcoming Improvements

* Add embedding pipelines for voice recordings, music files and videos.
* Provide graph‑analysis algorithms (centrality, shortest paths, etc.) through the API.
* Refine the search endpoints to better combine embeddings and metadata.

Contributions are welcome! Feel free to submit issues or pull requests with fixes and new features.
