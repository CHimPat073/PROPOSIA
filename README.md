# Proposia

Live Link:https://proposia-salesproposal.streamlit.app/

## AI-Powered RFP & Sales Proposal Copilot

> Turn client RFPs into grounded, professional sales proposals using
> your company's own knowledge base.

Proposia is a practical RAG-based sales engineering application that
helps businesses respond to Requests for Proposals (RFPs) faster and
more consistently.

Instead of asking an LLM to invent a proposal from scratch, Proposia
first processes the client's requirements, retrieves relevant
information from the company's private knowledge base, and uses that
grounded context to generate a proposal.

It supports both **PDF RFPs** and **plain-text RFPs**, with a
chatbot-style workflow for follow-up revisions.

------------------------------------------------------------------------

## Why Proposia?

A basic RAG demo often looks like:

``` text
PDF -> Chunk -> Embed -> Vector DB -> Question -> Answer
```

Proposia applies the same core concepts to a real business workflow:

``` text
Client RFP
   |
   +--> PDF / Text
           |
           v
      RFP Processing
           |
      Load -> Clean -> Chunk -> Embed
           |
           v
       Retriever
           |
           v
        ChromaDB
           |
           v
   Relevant Company Knowledge
           |
           v
       Prompt + Context
           |
           v
        Groq LLM
           |
           v
     Sales Proposal
           |
           v
       Chat / Refine
```

The project intentionally focuses on a small, understandable
architecture rather than unnecessary multi-agent complexity.

------------------------------------------------------------------------

# Problem Statement

Sales teams often receive long RFPs containing business, technical,
security, integration, delivery, timeline, and commercial requirements.

The team then has to manually:

1.  Understand the RFP.
2.  Find relevant company services.
3.  Search similar case studies.
4.  Identify technical capabilities.
5.  Find relevant technologies.
6.  Prepare an executive summary.
7.  Write the proposed solution.
8.  Prepare implementation details.
9.  Add pricing and assumptions.
10. Revise the proposal.

Proposia connects:

**Client requirements -\> Company knowledge -\> Retrieval -\> Grounded
proposal generation**

------------------------------------------------------------------------

# Features

-   PDF RFP processing
-   Plain-text RFP input
-   Company knowledge-base ingestion
-   Markdown, TXT, CSV, JSON and PDF knowledge sources
-   Metadata-aware document processing
-   Recursive chunking
-   Semantic embeddings
-   Persistent ChromaDB vector store
-   RFP-based semantic retrieval
-   Grounded proposal generation
-   Groq LLM integration
-   Chatbot-style proposal refinement
-   Modular Python architecture

------------------------------------------------------------------------

# Knowledge Base

The company knowledge base contains reusable business information:

``` text
knowledge_base/
|
+-- capabilities/
|   +-- ai_ml.md
|   +-- cloud.md
|   +-- data_engineering.md
|   +-- web_development.md
|
+-- case_studies/
|   +-- ecommerce.md
|   +-- fintech.md
|   +-- healthcare.md
|   +-- inventory.md
|
+-- company/
|   +-- company_profile.md
|   +-- services.md
|   +-- technology_stack.md
|
+-- pricing/
    +-- pricing.json
```

This knowledge can contain:

-   Company profile
-   Services
-   Technical capabilities
-   Industries
-   Case studies
-   Technology stack
-   Pricing

------------------------------------------------------------------------

# RAG Pipeline

## Knowledge-base ingestion

``` text
Company Documents
       |
       v
     Loader
       |
       v
   Metadata
       |
       v
    Chunker
       |
       v
   Embedder
       |
       v
    ChromaDB
```

## RFP processing

``` text
RFP PDF / Text
       |
       v
     Loader
       |
       v
     Parser
       |
       v
    Chunker
       |
       v
    Embedder
       |
       v
  RFP Query Chunks
       |
       v
    Retriever
       |
       v
    ChromaDB
       |
       v
Relevant Company Context
       |
       v
       LLM
       |
       v
Sales Proposal
```

An important architectural decision is that the **client RFP is not
permanently stored in the company's knowledge-base collection**.

The distinction is:

``` text
Company Knowledge
    -> Persistent information stored in ChromaDB

Client RFP
    -> Temporary input used to retrieve relevant company knowledge
```

------------------------------------------------------------------------

# Technology Stack

  Component             Technology
  --------------------- --------------------------------
  Language              Python
  RAG Framework         LangChain
  PDF Processing        PyPDFLoader
  Chunking              RecursiveCharacterTextSplitter
  Embeddings            BAAI/bge-small-en-v1.5
  Embedding Dimension   384
  Vector Database       ChromaDB
  Similarity Metric     Cosine
  LLM Provider          Groq
  LLM Model             Llama 3.3 70B Versatile
  Environment           python-dotenv
  API Layer             FastAPI (planned)
  Frontend              Chatbot UI (planned)

------------------------------------------------------------------------

# Architecture

``` text
                    +-------------------+
                    |     Client RFP     |
                    |   PDF / Text      |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |   RFP Processor   |
                    | Load / Parse       |
                    | Chunk / Embed      |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |     Retriever     |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |     ChromaDB      |
                    | Company Knowledge |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Relevant Context  |
                    +---------+---------+
                              |
                 +------------+------------+
                 |                         |
                 v                         v
            RFP Requirements        Proposal Rules
                 |                         |
                 +------------+------------+
                              |
                              v
                    +-------------------+
                    |    Groq / LLM      |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    |  Sales Proposal   |
                    +---------+---------+
                              |
                              v
                    +-------------------+
                    | Chat / Refinement |
                    +-------------------+
```

------------------------------------------------------------------------

# Project Structure

``` text
Proposia/
|
+-- backend/
|   +-- app/
|   |   +-- ingestion/
|   |   |   +-- __init__.py
|   |   |   +-- loader.py
|   |   |   +-- metadata.py
|   |   |   +-- chunker.py
|   |   |   +-- embedder.py
|   |   |
|   |   +-- database/
|   |   |   +-- __init__.py
|   |   |   +-- vector_store.py
|   |   |
|   |   +-- retrieval/
|   |   |   +-- __init__.py
|   |   |   +-- retriever.py
|   |   |
|   |   +-- rfp/
|   |   |   +-- __init__.py
|   |   |   +-- loader.py
|   |   |   +-- parser.py
|   |   |   +-- processor.py
|   |   |
|   |   +-- llm/
|   |       +-- __init__.py
|   |       +-- prompt.py
|   |       +-- client.py
|   |       +-- generator.py
|   |
|   +-- tests/
|       +-- test_full_pipeline.py
|
+-- knowledge_base/
|   +-- capabilities/
|   +-- case_studies/
|   +-- company/
|   +-- pricing/
|
+-- data/
|   +-- rfps/
|       +-- sample_rfp.pdf
|
+-- database/
|   +-- chroma/
|
+-- .env
+-- .gitignore
+-- requirements.txt
+-- README.md
```

------------------------------------------------------------------------

# Installation

## 1. Clone the repository

``` bash
git clone https://github.com/CHimPat073/PROPOSIA.git
cd Proposia
```

## 2. Create a virtual environment

Windows:

``` bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If required:

``` bash
pip install langchain-groq python-dotenv
```

------------------------------------------------------------------------

# Environment Variables

Create `.env` at the project root:

``` env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

Never commit the `.env` file.

Recommended `.gitignore` entries:

``` gitignore
.env
.venv/
__pycache__/
*.pyc
```

------------------------------------------------------------------------

# Running the Project

Run commands from the **project root**.

## Complete demo

``` bash
python -m backend.tests.test_full_pipeline
```

## PDF RFP

``` bash
python -m backend.tests.test_full_pipeline pdf
```

## Text RFP

``` bash
python -m backend.tests.test_full_pipeline text
```

## Interactive chatbot

``` bash
python -m backend.tests.test_full_pipeline chat
```

------------------------------------------------------------------------

# Example

### RFP input

``` text
We need a cloud-based inventory management platform
with predictive demand forecasting, real-time analytics,
multi-warehouse support and REST API integration.
```

### Retrieved company knowledge

``` text
cloud.md
inventory.md
data_engineering.md
ai_ml.md
case_studies/inventory.md
services.md
```

### Generated output

``` text
# Executive Summary

...

# Understanding of Requirements

...

# Proposed Solution

...

# Technical Approach

...

# Relevant Experience

...

# Implementation Timeline

...

# Pricing

...

# Next Steps

...
```

------------------------------------------------------------------------

# Chatbot Workflow

After the initial proposal is generated, the user can continue working
on the same RFP.

``` text
User:
Generate a proposal.

AI:
[Proposal]

User:
Make it shorter and emphasize security.

AI:
[Revised proposal]

User:
Add more detail about our cloud capabilities.

AI:
[Updated proposal]
```

This makes Proposia a lightweight **Sales Proposal Copilot**, rather
than only a document-question-answering application.

------------------------------------------------------------------------

# Design Principles

## 1. Separate knowledge from queries

The company knowledge base is persistent.

The client RFP is an input/query.

## 2. Reuse components

The system reuses the same embedding model and retrieval infrastructure
rather than creating separate pipelines for every input type.

## 3. Separation of responsibilities

``` text
loader.py
    -> Loading

parser.py
    -> Cleaning

chunker.py
    -> Chunking

embedder.py
    -> Embedding

vector_store.py
    -> Storage and similarity search

retriever.py
    -> Retrieval

prompt.py
    -> Prompt construction

client.py
    -> LLM communication

generator.py
    -> Proposal generation
```

## 4. Grounded generation

The proposal generator is instructed to rely on retrieved company
knowledge and avoid inventing unsupported company-specific information.

## 5. Keep the architecture explainable

The initial version intentionally avoids:

-   Multi-agent systems
-   LangGraph
-   BM25
-   Hybrid search
-   Reranking pipelines
-   Kafka
-   Kubernetes
-   Microservices

The objective is a practical and explainable RAG application.

------------------------------------------------------------------------

# Current Status

-   [x] Knowledge-base creation
-   [x] Multi-format document loading
-   [x] Metadata handling
-   [x] Text cleaning
-   [x] Recursive chunking
-   [x] Embedding pipeline
-   [x] ChromaDB persistence
-   [x] Vector similarity search
-   [x] RFP PDF loading
-   [x] RFP parsing
-   [x] RFP processing
-   [x] RFP embeddings
-   [x] RFP-based retrieval
-   [x] Groq LLM integration
-   [x] Prompt-based proposal generation
-   [x] Text RFP input
-   [x] PDF RFP input
-   [x] Initial chatbot workflow

### Planned

-   [ ] FastAPI API layer
-   [ ] Chatbot frontend
-   [ ] Proposal formatting
-   [ ] PDF export
-   [ ] Source/reference display
-   [ ] Retrieval and generation evaluation
-   [ ] Deployment

------------------------------------------------------------------------

# Future Improvements

Potential future extensions:

### RFP Intelligence

-   Automatic requirement extraction
-   Requirement categorization
-   Must-have vs nice-to-have detection
-   Risk and gap identification
-   Compliance matrix generation

### Retrieval

-   Hybrid semantic + keyword search
-   Reranking
-   Metadata filtering
-   Retrieval evaluation

### Proposal Intelligence

-   Proposal quality scoring
-   Requirement-to-proposal traceability
-   Missing requirement detection
-   Proposal versioning

### Business Integrations

-   CRM integration
-   Team collaboration
-   Approval workflows
-   Proposal analytics

These are intentionally outside the initial MVP.

------------------------------------------------------------------------

# What This Project Demonstrates

Proposia demonstrates practical understanding of:

-   RAG architecture
-   Data ingestion
-   Document processing
-   Chunking strategies
-   Embedding models
-   Vector databases
-   Semantic similarity search
-   Retrieval pipelines
-   Prompt engineering
-   LLM integration
-   Grounded generation
-   RFP processing
-   Conversational AI
-   Modular Python architecture
-   Real-world business workflow design

------------------------------------------------------------------------

# Project Vision

The long-term goal is not simply:

> Generate text with an LLM.

It is:

> **Help sales teams turn complex client requirements into accurate,
> company-grounded proposals faster.**

``` text
                 PROPOSIA
                    |
        +-----------+-----------+
        |           |           |
       RFP       Company      Sales
   Intelligence Knowledge   Intelligence
        |           |           |
        +-----------+-----------+
                    |
                    v
             Proposal Copilot
```

------------------------------------------------------------------------

# Author

**Himanshu Pathak**

Built as a learning-focused project to understand how RAG and LLM
systems can be applied to real-world business workflows.

------------------------------------------------------------------------

## Proposia

**From RFP to Proposal, grounded in your company's knowledge.**
