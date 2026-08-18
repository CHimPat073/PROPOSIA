"""# Proposia

## AI-Powered RFP & Sales Proposal Copilot

> Turn client RFPs into grounded, professional sales proposals using your company's own knowledge base.

Proposia is a practical RAG-based sales engineering application that helps businesses respond to Requests for Proposals (RFPs) faster and more consistently.

Instead of asking an LLM to invent a proposal from scratch, Proposia first processes the client's requirements, retrieves relevant information from the company's private knowledge base, and uses that grounded context to generate a proposal.

It supports both **PDF RFPs** and **plain-text RFPs**, with a chatbot-style workflow for follow-up revisions.

---

## Why Proposia?

A basic RAG demo often looks like:

```text
PDF -> Chunk -> Embed -> Vector DB -> Question -> Answer
