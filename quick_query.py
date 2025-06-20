from GraphReasoning import *
import os
import networkx as nx
import glob
from transformers import AutoTokenizer, AutoModel
from GraphReasoning.graph_tools import find_best_fitting_node_list
from GraphReasoning.graph_analysis import find_path_and_reason
from dotenv import load_dotenv

load_dotenv()

def load_graph_and_embeddings(output_dir="./ambroxol_output_v2/"):
    """Load the latest graph and embeddings."""
    # Load the specific integrated graph
    graph_file = os.path.join(output_dir, "graph_augmented_graphML_integrated.graphml")
    
    if not os.path.exists(graph_file):
        print(f"❌ Integrated graph file not found: {graph_file}")
        # Fallback to any graph file
        graph_files = glob.glob(os.path.join(output_dir, "*graph*.graphml"))
        if not graph_files:
            print(f"❌ No graph files found in {output_dir}")
            return None, None
        
        graph_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        graph_file = graph_files[0]
        print(f"⚠️ Using fallback graph: {graph_file}")
    else:
        print(f"✅ Using integrated graph: {graph_file}")
    
    try:
        G = nx.read_graphml(graph_file)
        print(f"✅ Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    except Exception as e:
        print(f"❌ Error loading graph: {e}")
        return None, None
    
    # Load embeddings
    embeddings_file = os.path.join(output_dir, "node_embeddings.pkl")
    embeddings = None
    if os.path.exists(embeddings_file):
        try:
            import pickle
            with open(embeddings_file, 'rb') as f:
                embeddings = pickle.load(f)
            print(f"✅ Loaded {len(embeddings)} embeddings")
        except Exception as e:
            print(f"⚠️ Error loading embeddings: {e}")
    
    return G, embeddings

def get_relationship_info(G, node1, node2):
    """Get comprehensive relationship information between two nodes."""
    info = {
        'node1': node1,
        'node2': node2,
        'direct_relationships': [],
        'shortest_paths': [],
        'common_neighbors': [],
        'node_stats': {}
    }
    
    # Direct relationships
    if G.has_edge(node1, node2):
        edge_data = G[node1][node2]
        info['direct_relationships'].append({
            'direction': 'forward',
            'relationship': edge_data.get('title', 'connected'),
            'weight': edge_data.get('weight', 1.0)
        })
    
    if G.has_edge(node2, node1):
        edge_data = G[node2][node1]
        info['direct_relationships'].append({
            'direction': 'reverse',
            'relationship': edge_data.get('title', 'connected'),
            'weight': edge_data.get('weight', 1.0)
        })
    
    # Shortest paths
    try:
        paths = list(nx.all_simple_paths(G, node1, node2, cutoff=4))
        paths.sort(key=len)
        for path in paths[:2]:  # Top 2 shortest paths
            path_info = {
                'length': len(path) - 1,
                'path': path,
                'edges': []
            }
            
            for i in range(len(path) - 1):
                u, v = path[i], path[i + 1]
                if G.has_edge(u, v):
                    edge_data = G[u][v]
                    path_info['edges'].append({
                        'from': u,
                        'to': v,
                        'relationship': edge_data.get('title', 'connected')
                    })
            
            info['shortest_paths'].append(path_info)
    except nx.NetworkXNoPath:
        pass
    
    # Common neighbors
    try:
        neighbors1 = set(G.neighbors(node1))
        neighbors2 = set(G.neighbors(node2))
        common = neighbors1.intersection(neighbors2)
        
        for neighbor in common:
            rel1 = G[node1][neighbor].get('title', 'connected') if G.has_edge(node1, neighbor) else 'unknown'
            rel2 = G[neighbor][node2].get('title', 'connected') if G.has_edge(neighbor, node2) else 'unknown'
            
            info['common_neighbors'].append({
                'neighbor': neighbor,
                'rel_to_node1': rel1,
                'rel_to_node2': rel2
            })
    except Exception as e:
        print(f"⚠️ Error finding common neighbors: {e}")
    
    # Node statistics
    info['node_stats'] = {
        'node1_degree': G.degree(node1),
        'node2_degree': G.degree(node2),
        'node1_neighbors': list(G.neighbors(node1))[:5],
        'node2_neighbors': list(G.neighbors(node2))[:5]
    }
    
    return info

def print_relationship_summary(info):
    """Print a formatted summary of the relationship."""
    print(f"\n🔍 Relationship Analysis: {info['node1']} ↔ {info['node2']}")
    print("=" * 60)
    
    # Direct relationships
    if info['direct_relationships']:
        print("🔗 Direct Relationships:")
        for rel in info['direct_relationships']:
            direction = "→" if rel['direction'] == 'forward' else "←"
            print(f"   {info['node1']} {direction} {info['node2']}: {rel['relationship']} (weight: {rel['weight']:.2f})")
    else:
        print("🔗 Direct Relationships: None found")
    
    # Shortest paths
    if info['shortest_paths']:
        print(f"\n🛤️ Shortest Paths ({len(info['shortest_paths'])} found):")
        for i, path_info in enumerate(info['shortest_paths'], 1):
            print(f"   Path {i} (length {path_info['length']}):")
            path_str = " → ".join(path_info['path'])
            print(f"      {path_str}")
            for edge in path_info['edges']:
                print(f"      {edge['from']} --[{edge['relationship']}]--> {edge['to']}")
    else:
        print("\n🛤️ Shortest Paths: No path found")
    
    # Common neighbors
    if info['common_neighbors']:
        print(f"\n🤝 Common Neighbors ({len(info['common_neighbors'])} found):")
        for neighbor_info in info['common_neighbors']:
            print(f"   • {neighbor_info['neighbor']}")
            print(f"     {info['node1']} --[{neighbor_info['rel_to_node1']}]--> {neighbor_info['neighbor']}")
            print(f"     {neighbor_info['neighbor']} --[{neighbor_info['rel_to_node2']}]--> {info['node2']}")
    else:
        print("\n🤝 Common Neighbors: None found")
    
    # Node statistics
    print(f"\n📊 Node Statistics:")
    print(f"   {info['node1']}: degree {info['node_stats']['node1_degree']}, neighbors: {info['node_stats']['node1_neighbors']}")
    print(f"   {info['node2']}: degree {info['node_stats']['node2_degree']}, neighbors: {info['node_stats']['node2_neighbors']}")

def generate_openai_llm(
    system_prompt="You are a biomaterials scientist.",
    prompt="What is ambroxol?",
    temperature=0.333,
    max_tokens=10000,
):
    # API Keys
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable:\n"
            "export OPENAI_API_KEY='your-api-key-here'"
        )

    try:
        from GraphReasoning.openai_tools import generate_OpenAIGPT
        response = generate_OpenAIGPT(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key,
            gpt_model="gpt-4o-mini",
            organization="",
            timeout=120,
            frequency_penalty=0,
            presence_penalty=0,
            top_p=1.0,
        )
        return response

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        print("Make sure your API key is valid and you have sufficient credits.")
        raise

def find_path_and_reason_analysis(G, embeddings, tokenizer, model, node1_query, node2_query, output_dir="./ambroxol_output_v2/"):
    """Use the find_path_and_reason function for detailed analysis with reasoning."""
    print(f"\n🧠 AI-Powered Path Analysis: {node1_query} ↔ {node2_query}")
    print("=" * 70)
    
    try:
        # Use the find_path_and_reason function from GraphReasoning
        response, (best_node_1, best_similarity_1, best_node_2, best_similarity_2), path, path_graph, shortest_path_length, fname, graph_GraphML = find_path_and_reason(
            G=G,
            node_embeddings=embeddings,
            tokenizer=tokenizer,
            model=model,
            generate=generate_openai_llm,
            keyword_1=node1_query,
            keyword_2=node2_query,
            include_keywords_as_nodes=True,
            inst_prepend="",
            graph_analysis_type="path and relations",
            instruction="Analyze the relationship between these concepts and explain how they are connected in the context of ambroxol research. What are the key insights about their relationship?",
            verbatim=False,
            N_limit=None,
            temperature=0.3,
            keywords_separator=" --> ",
            system_prompt="You are a biomedical researcher specializing in ambroxol and related therapeutic mechanisms. Analyze the relationships logically and provide scientific insights.",
            max_tokens=2048,
            prepend="You are analyzing relationships in a knowledge graph about ambroxol and related biomedical concepts. The graph shows connections between molecules, diseases, and biological processes.\n\n",
            similarity_fit_ID_node_1=0,
            similarity_fit_ID_node_2=0,
            save_files=True,
            data_dir=output_dir,
            visualize_paths_as_graph=True,
            display_graph=False,
            words_per_line=2,
        )
        
        print(f"\n✅ AI Analysis Complete!")
        print(f"📊 Path Statistics:")
        print(f"   • Path length: {shortest_path_length}")
        print(f"   • Nodes in path: {len(path)}")
        print(f"   • Best match for '{node1_query}': {best_node_1} (similarity: {best_similarity_1:.3f})")
        print(f"   • Best match for '{node2_query}': {best_node_2} (similarity: {best_similarity_2:.3f})")
        
        if fname:
            print(f"📁 Visualization saved: {fname}")
        if graph_GraphML:
            print(f"📁 GraphML saved: {graph_GraphML}")
        
        return {
            'response': response,
            'path': path,
            'path_length': shortest_path_length,
            'best_matches': (best_node_1, best_similarity_1, best_node_2, best_similarity_2),
            'visualization': fname,
            'graphml': graph_GraphML
        }
        
    except Exception as e:
        print(f"❌ Error in AI path analysis: {e}")
        return None

def quick_query(node1_query, node2_query, output_dir="./ambroxol_output_v2/"):
    """Quick query function for programmatic use."""
    print(f"🔍 Quick Query: {node1_query} ↔ {node2_query}")
    
    # Load graph and embeddings
    G, embeddings = load_graph_and_embeddings(output_dir)
    if G is None:
        print("❌ Could not load graph")
        return None
    
    # Initialize embedding model
    try:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except Exception as e:
        print(f"❌ Could not initialize embedding model: {e}")
        return None
    
    # Use AI-powered analysis
    ai_result = find_path_and_reason_analysis(G, embeddings, tokenizer, model, node1_query, node2_query, output_dir)
    
    return ai_result

def run_example_queries():
    """Run some example queries."""
    queries = [
        ("ambroxol", "GCase"),
        ("ambroxol", "alpha synuclein"),
        ("ambroxol", "dopamine receptor"),
        ("ambroxol", "TFEB"),
    ]
    
    print("🧪 Running All Example Queries with AI Analysis")
    print("=" * 60)
    
    results = []
    for i, (node1, node2) in enumerate(queries, 1):
        print(f"\n{'='*20} Query {i}/{len(queries)}: {node1} ↔ {node2} {'='*20}")
        try:
            result = quick_query(node1, node2)
            if result is not None:
                results.append({
                    'query': (node1, node2),
                    'result': result,
                    'status': 'success'
                })
                print(f"✅ Query {i} completed successfully")
            else:
                results.append({
                    'query': (node1, node2),
                    'result': None,
                    'status': 'failed'
                })
                print(f"❌ Query {i} failed")
        except Exception as e:
            print(f"❌ Query {i} failed with error: {e}")
            results.append({
                'query': (node1, node2),
                'result': None,
                'status': 'error',
                'error': str(e)
            })
        print()
    
    # Summary
    print("📊 QUERY SUMMARY")
    print("=" * 50)
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] in ['failed', 'error']]
    
    print(f"✅ Successful queries: {len(successful)}/{len(queries)}")
    print(f"❌ Failed queries: {len(failed)}/{len(queries)}")
    
    if successful:
        print(f"\n📋 Successful Relationships Found:")
        for result in successful:
            node1, node2 = result['query']
            ai_result = result['result']
            path_length = ai_result.get('path_length', 'unknown')
            print(f"   • {node1} ↔ {node2}: path length {path_length}")
    
    if failed:
        print(f"\n❌ Failed Queries:")
        for result in failed:
            node1, node2 = result['query']
            status = result['status']
            error = result.get('error', 'No specific error')
            print(f"   • {node1} ↔ {node2}: {status} - {error}")
    
    return results

if __name__ == "__main__":
    # You can either run example queries or do a specific query
    import sys
    
    if len(sys.argv) == 3:
        # Command line usage: python quick_query.py "node1" "node2"
        node1_query = sys.argv[1]
        node2_query = sys.argv[2]
        print(f"🔍 Running specific query: {node1_query} ↔ {node2_query}")
        quick_query(node1_query, node2_query)
    else:
        # Run all example queries by default
        print("🚀 Starting comprehensive relationship analysis with AI reasoning...")
        results = run_example_queries()
        print(f"\n🎉 Analysis complete! Processed {len(results)} queries.") 