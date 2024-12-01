import pickle
import random
import re

import networkx as nx
import numpy as np
from colorama import Fore, Style
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from pyvis.network import Network

from prompts import entity_relation_extraction
from prompts.defaults import completion_delimiter, entity_types, tuple_delimiter, record_delimiter
from prompts.summarize_entity_relation_descriptions import SUMMARIZE_ENTITIES


class KnowledgeGraph:
    def __init__(self, graphml_path: str, llm_model: str, llm_embedding: str):
        self.graphml_path = graphml_path
        self.llm = OllamaLLM(model=llm_model)
        self.llm_embedding = OllamaEmbeddings(model=llm_embedding)
        self.graph = nx.Graph()
        self.vector_db = {}  # A dictionary to store vectors and corresponding nodes/edges

    def build_from_file(self, file_name: str):
        with open(file_name, 'r') as file:
            lines = file.read().split('\n')
            pattern = re.compile(r'^\s*\(.*\)\s*$')
            lines = [line for line in lines if pattern.match(line)]
            for line in lines:
                self._process_line(line)
        self._clean_graph(self.graph)
        nx.write_graphml(self.graph, self.graphml_path)

    def _process_line(self, line: str):
        line = re.sub(r"^\(|\)$", "", line.strip())
        attributes = line.split(tuple_delimiter)

        if "entity" in attributes[0] and len(attributes) >= 4:
            self._add_entity(attributes)
        elif "relationship" in attributes[0] and len(attributes) >= 4:
            self._add_relationship(attributes)

    def _add_entity(self, attributes: list):
        entity_name = attributes[1].upper()
        entity_type = attributes[2].upper()
        entity_description = attributes[3].upper()

        if entity_name in self.graph.nodes:
            node = self.graph.nodes[entity_name]
            node["description"] = "\n".join([node.get("description", ""), entity_description])
            node["type"] = entity_type if entity_type else node["type"]
        else:
            self.graph.add_node(entity_name, type=entity_type, description=entity_description, source_id=0)

    def _add_relationship(self, attributes: list):
        source = attributes[1].upper()
        target = attributes[2].upper()
        description = attributes[3].upper()
        weight = self._extract_weight(attributes)

        if source not in self.graph.nodes:
            self.graph.add_node(source, type="", description="", source_id=0)
        if target not in self.graph.nodes:
            self.graph.add_node(target, type="", description="", source_id=0)

        if self.graph.has_edge(source, target):
            edge_data = self.graph.get_edge_data(source, target)
            if edge_data:
                weight += edge_data["weight"]
                description = "\n".join([edge_data["description"], description])
        self.graph.add_edge(source, target, weight=weight, description=description, source_id=0)

    @staticmethod
    def _extract_weight(attributes: list):
        try:
            return float(attributes[-1])
        except ValueError:
            return 1.0

    def _clean_graph(self, graph):
        for node, data in self.graph.nodes(data=True):
            original_description = data.get("description", "")
            if original_description and "\n" in original_description:
                formatted_prompt = SUMMARIZE_ENTITIES.format(
                    entity_name=node,
                    description_list=original_description.split("\n"),
                )
                print("old desc:")
                print(original_description)

                response = self.llm.invoke(formatted_prompt)

                print("   cleaned desc:")
                print(f"   {response}")
                data["description"] = response

    def build_vectordb(self, vector_db_path="output/vector_db.pkl"):
        # Loop over nodes and relationships to embed them and store in vector DB
        print("Building vector database...")

        if not self.graph:
            self.graph = nx.read_graphml(self.graphml_path)

        for node, data in self.graph.nodes(data=True):
            description = data.get("description", "")
            if not description:
                description = node
            vector = self._get_embedding(description)
            self.vector_db[node] = {"vector": vector, "type": "node", "data": data}

        for source, target, data in self.graph.edges(data=True):
            relationship_description = data.get("description", "")
            if not relationship_description:
                relationship_description = f"{source} {target}"
            vector = self._get_embedding(relationship_description)
            edge_key = (source, target)
            self.vector_db[edge_key] = {"vector": vector, "type": "edge", "data": data}

            with open(vector_db_path, "wb") as f:
                pickle.dump(self.vector_db, f)

        print("Vector database built successfully.")

    def query_graph(self, query: str):
        if not self.vector_db:
            with open("output/vector_db.pkl", "rb") as f:
                self.vector_db = pickle.load(f)

        query_vector = self._get_embedding(query)
        nearest_nodes, nearest_edges = self._find_nearest_vectors(query_vector)
        # TODO: this can better, we can also retrieve the neighbours of the nearest nodes or find a path between nodes
        result = self._generate_answer_from_nearest(nearest_nodes, nearest_edges)

        return result

    def _get_embedding(self, text: str):
        # Use the LLM embedding model to generate an embedding for the text
        response = self.llm_embedding.embed_query(text)
        # Assuming the response is a list of floats (embedding vector)
        return np.array(response)  # Convert response to numpy array for easier comparison

    def _find_nearest_vectors(self, query_vector: np.array, top_n=5):
        # Compute cosine similarity between query vector and all vectors in the vector database
        similarities = {}
        for key, value in self.vector_db.items():
            vector = value["vector"]
            if vector is not None:
                similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
                similarities[key] = similarity

        # Sort the items by similarity in descending order
        sorted_items = sorted(similarities.items(), key=lambda item: item[1], reverse=True)

        # Retrieve top_n closest nodes and edges
        nearest_nodes = []
        nearest_edges = []
        for item in sorted_items[:top_n]:
            key, similarity = item
            if self.vector_db[key]["type"] == "node":
                nearest_nodes.append((key, self.vector_db[key]["data"], similarity))
            elif self.vector_db[key]["type"] == "edge":
                nearest_edges.append((key, self.vector_db[key]["data"], similarity))

        return nearest_nodes, nearest_edges

    def _generate_answer_from_nearest(self, nearest_nodes, nearest_edges):
        # Combine the nearest nodes and edges into a text-based response
        result = "Based on the query, the following information was retrieved:\n"

        result += "\nNearest Nodes:\n"
        for node, data, similarity in nearest_nodes:
            result += f"Node: {node}\nType: {data['type']}\nDescription: {data.get('description', 'No description available')}\nSimilarity: {similarity}\n\n"

        result += "\nNearest Edges:\n"
        for edge, data, similarity in nearest_edges:
            result += f"Edge: {edge[0]} -> {edge[1]}\nDescription: {data.get('description', 'No description available')}\nSimilarity: {similarity}\n\n"

        return result

    def show(self, output_file="output/knowledge_graph.html"):
        def insert_newlines(text):
            return re.sub(r'(\. )', r'\1\n', text).strip("\n")

        # Example usage
        net = Network(height="100vh", notebook=True)
        net.from_nx(self.graph)

        for node in net.nodes:
            node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
            node["title"] = f"Name: {node['id']}\nDescription: {insert_newlines(node.get('description', 'N/A'))}"

        for edge in net.edges:
            edge_data = self.graph.get_edge_data(edge["from"], edge["to"])
            if edge_data:
                description = edge_data.get("description", "N/A")
                weight = max(edge_data.get("weight", 1.0),
                             edge_data.get("width", 1.0))  # For some reason the library changes weight to width?
                edge["title"] = f"Description: {insert_newlines(description)}\nWeight: {weight}"

        print(f"Open {output_file} in your browser to view the knowledge graph in html format.")
        net.show(output_file)


class TextProcessor:
    # TODO: there are better ways to chunk text, this is just a simple example
    @staticmethod
    def split_text_into_chunks(text: str, max_chunk_size=2000, overlap=250) -> list:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chunk_size, len(text))
            chunks.append(text[start:end])
            start += max_chunk_size - overlap
        return chunks


class EntityExtractor:
    def __init__(self, llm_model: str):
        self.llm = OllamaLLM(model=llm_model)

    def extract_entities_from_file(self, input_file_path: str,
                                   output_file_path: str = "output/extracted_entities_relations.txt") -> list:
        with open(input_file_path, 'r') as file:
            text = file.read()

        chunks = TextProcessor.split_text_into_chunks(text)
        prompt = PromptTemplate(
            template=entity_relation_extraction.GRAPH_EXTRACTION_PROMPT,
            input_variables=["entity_types", "tuple_delimiter", "record_delimiter", "completion_delimiter",
                             "input_text"]
        )

        responses = []

        for i, chunk in enumerate(chunks):
            print(f"    Processing chunk {i + 1} of {len(chunks)}")
            formatted_prompt = prompt.format(
                entity_types=entity_types,
                tuple_delimiter=tuple_delimiter,
                record_delimiter=record_delimiter,
                completion_delimiter=completion_delimiter,
                input_text=chunk
            )
            dirty_responses = self._extract_entities_with_loop(formatted_prompt)
            print(dirty_responses)
            cleaned_responses = [item.strip(" \n") for r in dirty_responses if "entity" in r or "relationship" in r for
                                 item in
                                 r.strip(" \n").split(record_delimiter)]
            # cleaned_responses = self._cleanup_responses(dirty_responses)
            print(cleaned_responses)
            responses.extend(cleaned_responses)

        with open(output_file_path, "w") as f:
            f.write("\n".join(responses))
        return responses

    def _extract_entities_with_loop(self, formatted_prompt: str) -> list:
        response = self.llm.invoke(formatted_prompt)
        responses = [response]

        while completion_delimiter not in response:
            next_prompt = f"{response}\n\n{entity_relation_extraction.CONTINUE_PROMPT}"
            response = self.llm.invoke(next_prompt)
            responses.append(response)

            check_prompt = f"{response}\n\n{entity_relation_extraction.LOOP_PROMPT}"
            if self.llm.invoke(check_prompt).strip().lower() != "yes":
                break

        return responses

    # def _cleanup_responses(self, responses: list[str]) -> list:
    #
    #     print("----In CLEANUP RESPONSES----")
    #     print("input:")
    #     print(responses)
    #
    #     cleaned_responses = []
    #     for r in responses:
    #         formatted_prompt = entity_relation_extraction.CLEANUP_PROMPT.format(
    #             entity_types=entity_types,
    #             tuple_delimiter=tuple_delimiter,
    #             record_delimiter=record_delimiter,
    #             completion_delimiter=completion_delimiter,
    #             data=r.split(record_delimiter)
    #         )
    #         cleaned_r = self.llm.invoke(formatted_prompt)
    #         cleaned_responses.append(cleaned_r)
    #
    #
    #     print("CLEANED UP:")
    #     print(cleaned_responses)
    #     print("------")
    #     return cleaned_responses


class Pipeline:
    def __init__(self, text_file: str, llm_model: str, llm_embedding: str):
        self.text_file = text_file
        self.llm_model = llm_model
        self.entity_extractor = EntityExtractor(llm_model)
        self.graph = KnowledgeGraph("output/knowledge_graph.graphml", llm_model, llm_embedding)

    def run(self):
        print(Fore.YELLOW + "STARTING PIPELINE...")

        print(Fore.GREEN + "Extracting entities and relationships from text...")
        print(Style.RESET_ALL)
        self.entity_extractor.extract_entities_from_file(self.text_file, "output/response.txt")

        print(Fore.GREEN + "Building the knowledge graph...")
        print(Style.RESET_ALL)
        self.graph.build_from_file("output/response.txt")

        print(Fore.GREEN + "VISUALIZING THE KNOWLEDGE GRAPH...")
        print(Style.RESET_ALL)

        self.graph.show()

        print(Fore.GREEN + "Building the vector database...")
        print(Style.RESET_ALL)
        self.graph.build_vectordb()

        print(Fore.GREEN + "Querying the knowledge graph...")
        print(Style.RESET_ALL)
        answer = self.graph.query_graph("Who earns £15 per week and who is his or her boss?")
        print(answer)  # TODO: dit moet dan eigenlijk nog door een LLM gaan voor NLP output

        print(Fore.GREEN + "Pipeline completed!")


if __name__ == "__main__":
    text_file_path = "data/a_christmas_carol.txt"
    ollama_model = "qwen2"
    embedding_model = "nomic-embed-text"

    pipeline = Pipeline(text_file_path, ollama_model, embedding_model)
    pipeline.run()
