# README

Since 2022, large language models (LLMs) have surged in popularity due to their ability to generate human-like text and serve as powerful tools for natural language processing (NLP).
But soon, it became an issue that LLMs could not generate text or reason on private or new data, or worse: hallucinate. 

Retrieval-Augmented Generation (RAG) emerged as a promising solution to these problems. But RAG has some issues as well.
* RAG neglects relationships when information is isolated in different text chunks.
* RAG systems may contain a lot of redundant information, which can lead to a "lost in the middle" problem.
* RAG can only focus on a subset of documents and could fail to comprehend the broader context.

GraphRAG aims to mitigate these issues by enriching an LLM with a graph-based knowledge base.

This repository contains the code for a basic GraphRAG implementation. 
While it is not as sophisticated as [Microsoft's implementation of GraphRAG](https://github.com/microsoft/graphrag) or [Hong Kong University's LightRAG](https://github.com/HKUDS/LightRAG), 
this repository could serve as
* a starting point for learning about GraphRAG
* a baseline when building and/or testing more complex implementations


## How does this implementation work?
* Extract entities and relationships from a text using an LLM
* Clean them up and store them in a graph
* Store every node, relationship and description of nodes and relations in a vector database
* Query the vector database when inputting a question for the LLM
* Retrieve relevant network components and feed them to an LLM to generate an answer


## How to run?

* Install the necessary dependencies, e.g. by running the commands:
```bash
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```
* You will also need to install [Ollama](https://ollama.com/), a tool to get up and running with Large Language Models on a local machine.
```bash
$ curl -fsSL https://ollama.com/install.sh | sh
$ ollama pull qwen2.5:14b  # or any other model you wish to use
```