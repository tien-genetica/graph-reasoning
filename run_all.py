from GraphReasoning import *
import os
from GraphReasoning.openai_tools import generate_OpenAIGPT
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import glob
import networkx as nx
from dotenv import load_dotenv

load_dotenv()

queries = [
    ("ambroxol", "GCase", "we expect ambroxol to increase activity of GCase"),
    (
        "ambroxol",
        "alpha synuclein",
        "we expect ambroxol to increase activity of alpha synuclein",
    ),
    (
        "ambroxol",
        "dopamine receptor",
        "we expect ambroxol to decrease activity of dopamine receptor",
    ),
    ("ambroxol", "TFEB", "we expect ambroxol to increase activity of TFEB"),
]


def process(text: str) -> str:
    """
    Extracts the main content of a research paper, removing metadata like authors, dates,
    affiliations, journal info, acknowledgments, and references.

    Args:
        text: The raw text read from the paper.

    Returns:
        A string containing only the core content of the paper.
        Returns an empty string if no core content is identified.
    """
    system_prompt = (
        "You are an expert academic parser. Your task is to extract only the core "
        "content of a research paper from the given raw text. "
        "This includes the Abstract, Introduction, Methods, Results, Discussion, "
        "Conclusion, and any other sections that contribute to the scientific "
        "narrative and findings. "
        "Strictly exclude all metadata and non-content elements such as: "
        "Authors, Affiliations, Author Contributions, Corresponding Author details, "
        "Dates (Received, Accepted, Published), Journal Title, Volume, Issue, Pages, "
        "DOI, ISSN, Copyright information, Funding statements, Acknowledgments, "
        "References, Bibliography, Appendices, Figure Captions (unless they are "
        "integrated into the main text in a way that requires their presence for understanding), "
        "Table of Contents, or any headers/footers."
        "If the text appears to contain no scientific content, return 'NO_CONTENT'."
    )

    prompt = f"Please extract the main body content from the following raw paper text:\n\n{text}\n\nExtracted Content:"

    extracted_content = generate_openai_llm(
        system_prompt=system_prompt,
        prompt=prompt,
        temperature=0.1,  # Keep temperature low for precise extraction
        max_tokens=10000, # Use a high max_tokens, as paper content can be long
    )

    if extracted_content.strip() == "NO_CONTENT":
        return ""
    
    # Some post-processing might be needed if the LLM adds introductory phrases
    # For example, if it says "Here is the extracted content:"
    # You might want to remove such phrases.
    # A common pattern is to just return the LLM's output directly, and trust
    # its instruction to only return the content.
    
    return extracted_content.strip()



# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize embedding model and tokenizer
def initialize_embedding_model():
    """Initialize the embedding model and tokenizer for node embeddings."""
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
        return process(text)
    except Exception as e:
        print(f"Error processing {os.path.basename(pdf_path)}: {e}")
        return ""

def generate_openai_llm(
    system_prompt="You are a biomaterials scientist.",
    prompt="What is ambroxol?",
    temperature=0.333,
    max_tokens=10000,
):
    if not openai_api_key:
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable:\n"
            "export OPENAI_API_KEY='your-api-key-here'"
        )

    try:
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

def run():
    try:
        # Import required functions
        from GraphReasoning.graph_generation import make_graph_from_text, add_new_subgraph_from_text
        from GraphReasoning.graph_tools import generate_node_embeddings

        print("🚀 Starting knowledge graph generation from all PDF files...")
        
        # Initialize embedding model
        print("📚 Initializing embedding model...")
        tokenizer, model = initialize_embedding_model()
        
        # Get all PDF files in test-ambroxol directory
        pdf_files = glob.glob("test-ambroxol/*.pdf")
        pdf_files.sort()  # Sort for consistent processing order
        
        if not pdf_files:
            print("❌ No PDF files found in test-ambroxol directory")
            return
            
        print(f"📄 Found {len(pdf_files)} PDF files to process")
        
        # Create output directory
        output_dir = "./ambroxol_output_v2/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process first PDF to create initial graph
        print(f"\n📖 Processing first PDF: {os.path.basename(pdf_files[0])}")
        first_pdf_text = extract_text_from_pdf(pdf_files[0])
        
        if not first_pdf_text.strip():
            print(f"⚠️ No text extracted from {pdf_files[0]}, skipping...")
            return
            
        # Create initial graph from first PDF
        graph_HTML, graph_GraphML, G, net, output_pdf = make_graph_from_text(
            txt=first_pdf_text,
            generate=generate_openai_llm,
            include_contextual_proximity=False,
            graph_root="ambroxol_graph_initial",
            chunk_size=500,
            chunk_overlap=50,
            repeat_refine=0,
            verbatim=False,
            data_dir=output_dir,
            save_HTML=True,
        )
        
        print(f"✅ Initial graph created from {os.path.basename(pdf_files[0])}")
        print(f"📊 Initial graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
        
        # Generate embeddings for the initial graph
        print("🔍 Generating embeddings for initial graph...")
        node_embeddings = generate_node_embeddings(G, tokenizer, model)
        
        # Process remaining PDFs and add to the graph
        for i, pdf_file in enumerate(pdf_files[1:], 1):
            print(f"\n📖 Processing PDF {i+1}/{len(pdf_files)}: {os.path.basename(pdf_file)}")
            
            pdf_text = extract_text_from_pdf(pdf_file)
            
            if not pdf_text.strip():
                print(f"⚠️ No text extracted from {pdf_file}, skipping...")
                continue
                
            try:
                # Add new subgraph from this PDF
                graph_GraphML_new, G_new, G_loaded, G_original, node_embeddings, res = add_new_subgraph_from_text(
                    txt=pdf_text,
                    generate=generate_openai_llm,
                    node_embeddings=node_embeddings,
                    tokenizer=tokenizer,
                    model=model,
                    original_graph_path_and_fname=graph_GraphML,
                    data_dir_output=output_dir,
                    verbatim=False,
                    size_threshold=10,
                    chunk_size=500,
                    do_Louvain_on_new_graph=True,
                    include_contextual_proximity=False,
                    repeat_refine=0,
                    similarity_threshold=0.95,
                    do_simplify_graph=True,
                    return_only_giant_component=False,
                    save_common_graph=True,
                    max_workers=4,
                )
                
                # Update the main graph and embeddings
                G = G_new
                graph_GraphML = graph_GraphML_new
                
                print(f"✅ Added subgraph from {os.path.basename(pdf_file)}")
                print(f"📊 Updated graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
                
            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                continue
        
        # Final statistics
        print(f"\n🎉 Final knowledge graph generation completed!")
        print(f"📊 Final graph statistics:")
        print(f"   • Nodes: {G.number_of_nodes()}")
        print(f"   • Edges: {G.number_of_edges()}")
        print(f"   • Connected components: {nx.number_connected_components(G)}")
        print(f"   • Files saved to: {output_dir}")
        
        # Save final graph statistics
        if res:
            print(f"📈 Graph analysis results: {res}")

    except ImportError:
        print("❌ GraphReasoning not found. Make sure it's properly installed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run()
