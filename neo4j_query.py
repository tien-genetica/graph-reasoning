import sys
import os
from dotenv import load_dotenv

try:
    from neo4j import GraphDatabase
except ImportError:
    print("The 'neo4j' package is not installed. Please run: pip install neo4j")
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("The 'sentence-transformers' package is not installed. Please run: pip install sentence-transformers")
    sys.exit(1)

load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Your example queries
queries = [
    ("ambroxol", "GCase"),
    ("ambroxol", "alpha synuclein"),
    ("ambroxol", "dopamine receptor"),
    ("ambroxol", "TFEB"),
]

def get_node_display_name(node):
    """Gets a display name for a node, trying 'text', 'name', then 'id' properties."""
    display_name = node.get('text') or node.get('name') or node.get('id')
    if display_name:
        return display_name
    # As a fallback for debugging, show all node properties except embeddings
    node_dict = dict(node)
    if 'embedding' in node_dict:
        del node_dict['embedding']
    return str(node_dict)

def get_embedding(text, model):
    emb = model.encode([text])
    return emb[0].tolist()

def find_entity_by_vector(tx, embedding):
    # Try to find an entity node directly using the 'vector' index
    try:
        cypher = """
        CALL db.index.vector.queryNodes('vector', 1, $embedding)
        YIELD node, score
        RETURN elementId(node) AS node_eid, labels(node) AS labels, node, score
        ORDER BY score DESC
        LIMIT 1
        """
        result = tx.run(cypher, embedding=embedding)
        return result.single()
    except Exception as e:
        print(f"Error using 'vector' index: {e}")
        return None

def find_entity_via_chunk(tx, embedding):
    # Fallback: find a Chunk node, then traverse PART_OF to the entity
    try:
        cypher = """
        CALL db.index.vector.queryNodes('abstract-embeddings', 1, $embedding)
        YIELD node AS chunkNode, score
        MATCH (chunkNode)-[:PART_OF]-(entityNode)
        RETURN elementId(entityNode) AS node_eid, labels(entityNode) AS labels, entityNode, score
        ORDER BY score DESC
        LIMIT 1
        """
        result = tx.run(cypher, embedding=embedding)
        return result.single()
    except Exception as e:
        print(f"Error using 'abstract-embeddings' index: {e}")
        return None

def get_entity_for_query(tx, embedding):
    # Try direct entity search first, then fallback to chunk traversal
    entity = find_entity_by_vector(tx, embedding)
    if entity:
        return entity
    return find_entity_via_chunk(tx, embedding)

def find_shortest_path(tx, node_eid1, node_eid2):
    cypher = """
    MATCH (start), (end)
    WHERE elementId(start) = $eid1 AND elementId(end) = $eid2
    MATCH path = shortestPath((start)-[*..15]-(end))
    RETURN path
    """
    result = tx.run(cypher, eid1=node_eid1, eid2=node_eid2)
    record = result.single()
    if record:
        return record["path"]
    return None

def run_query(query1, query2, model, driver):
    with driver.session(database=NEO4J_DATABASE) as session:
        embedding1 = get_embedding(query1, model)
        embedding2 = get_embedding(query2, model)

        node1_info = session.execute_read(get_entity_for_query, embedding1)
        node2_info = session.execute_read(get_entity_for_query, embedding2)

        if not node1_info:
            print(f"❌ Could not find a related entity for '{query1}'.")
            return
        
        if not node2_info:
            print(f"❌ Could not find a related entity for '{query2}'.")
            return

        print(f"🔎 Found entity for '{query1}' (score {node1_info['score']:.3f}):")
        print(f"  - Node:   \"{get_node_display_name(node1_info['node'])}\"")
        print(f"  - Labels: {node1_info['labels']}")
        
        print(f"\n🔎 Found entity for '{query2}' (score {node2_info['score']:.3f}):")
        print(f"  - Node:   \"{get_node_display_name(node2_info['node'])}\"")
        print(f"  - Labels: {node2_info['labels']}")

        # Find shortest path between the two nodes
        path = session.execute_read(find_shortest_path, node1_info['node_eid'], node2_info['node_eid'])
        if path:
            print("\n🔗 Shortest path found:")
            
            nodes = path.nodes
            relationships = path.relationships
            
            path_str_parts = [f"({get_node_display_name(nodes[0])})"]
            
            for i, rel in enumerate(relationships):
                end_node_name = get_node_display_name(nodes[i+1])
                path_str_parts.append(f"-[{rel.type}]->({end_node_name})")
            
            print("  " + "".join(path_str_parts))
        else:
            print("\n❌ No path found between the two nodes.")


def batch_test_queries():
    print("=== Batch Neo4j Vector Search ===")
    # Load embedding model once
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return
    # Connect to Neo4j
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
        )
        driver.verify_connectivity()
    except Exception as e:
        print(f"Error connecting to Neo4j: {e}")
        return
    for q1, q2 in queries:
        print(f"\n{'='*10} Query: {q1} ↔ {q2} {'='*10}")
        try:
            run_query(q1, q2, model, driver)
        except Exception as e:
            print(f"Error running query for '{q1}' and '{q2}': {e}")
    driver.close()

if __name__ == "__main__":
    batch_test_queries()
