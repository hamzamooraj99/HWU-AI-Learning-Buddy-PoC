# AI Learning Buddy: RAG Proof of Concept
This repository contains a Proof of Concept (PoC) for an AI-powered "Learning Buddy" designed to assist university students by providing quick and accurate answers to questions based on their course documents. The system leverages a Retrieval-Augmented Generation (RAG) architecture to ground a local Large Language Model (LLM) in specific, factual information from course materials.
The entire system is containerized using Docker, allowing for a portable and reproducible deployment on any compatible machine.

## Features
- **RAG Architecture:** Combines a vector database (Milvus) for information retrieval with a language model (Ollama) for text generation, ensuring responses are factual and relevant to the provided documents.
- **Course-Specific Queries:** Supports multiple courses by isolating data based on a `course_id` within the vector database.
- **Local-First Design:** Utilizes an open-source LLM (Llama 3) running locally via Ollama, eliminating the need for paid API services and addressing data privacy concerns.

## System Architecture
The system is designed as a set of interconnected services orchestrated by Docker Compose, running on a single host machine.
- **Application Container (app.py):** Runs the Streamlit web application. It is the user-facing component and acts as the orchestrator of the entire RAG (“Retrieval-Augmented Generation”) pipeline.
- **LLM Container (ollama_llm):** Hosts the Llama3 LLM model using Ollama. It receives augmented prompts from the application container and generates responses.
- **Vector Database Container (milvus_db):** A Milvus instance that stores and provides high-speed vector search for the course document embeddings.
- **Persistent Volumes:** Docker volumes (milvus_data, ollama_models) ensure that the vector database and the LLM models are preserved across container restarts.
- **User Interface:** A web browser through which the user interacts with the Streamlit app.

![alt text](https://github.com/hamzamooraj99/HWU-AI-Learning-Buddy-PoC/blob/main/notebooks/sys_architecture.png)

## Workflow
The following workflow diagram illustrates the sequence of events when a user submits a query:  
![alt text](https://github.com/hamzamooraj99/HWU-AI-Learning-Buddy-PoC/blob/main/notebooks/diagram-export-9-5-2025-2_43_55-PM.png)

# Getting Started
## Prerequisites
To run this project, you need to have Docker and Docker Compose installed on your machine.
## Deployment
### 1. Clone the repository
```bash
git clone https://github.com/hamzamooraj99/HWU-AI-Learning-Buddy-PoC/
cd HWU-AI-Learning-Buddy-PoC
```
### 2. Start the services
This command will download the necessary Docker images, build the application container, and start all services in detached mode.
```bash
docker-compose up -d
```
This will spin up three services: `milvus_db`, `ollama_llm`, and `app`.

## Data Ingestion & Indexing
**Note:** This is a manual, offline process that must be completed before the application can function. It converts your course documents into a searchable format for the RAG system.
### 1. Data Ingestion
- Place your PDF course documents inside a directory, `./pdfs/[COURSE_ID]`
- Run the ingestion script to convert PDFs to markdown text, chunk the markdown text via headings, and parse them into a JSON format
```bash
python ./src/1_ingest_data.py
```
### 2. Generate Embeddings
- This script will read the JSON data from the previous step and generate vector embeddings using the `BAAI/bge-small-en-v1.5` model
```bash
python src/2_gen_embeddings.py
```
### 3. Vector Indexing
- This final script connects to the running Milvus DB and inserts the documents and their corresponding embeddings, making them searchable
```bash
python src/3_vector_indexing.py
```

## Usage
Once all the services are running and the data has been indexed, the Streamlit application will be accessible.
- Open your web browser and navigate to: `[PENDING]`
You can now select a course and begin asking questions based on the documents you indexed.


# Limitations & Future Work
This is a proof of concept with known limitations, primarily in its current single-server, single-user design. Future work would focus on:
- **Scalability:** Migrating to a distributed architecture using Kubernetes and a managed vector database service like Milvus Distributed.
- **Automated Data Pipeline:** Developing an automated system to regularly ingest and update course information from platforms like Canvas using API tokens, eliminating the current manual process.
- **Enhanced Models:** Exploring the use of larger, more powerful embedding models and a transition to cloud-based LLM services for improved performance and reliability.
- **Personalization:** Integrating with student profiles to provide personalized support (e.g., study plans, grade tracking).
