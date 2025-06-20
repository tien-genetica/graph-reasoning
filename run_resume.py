from GraphReasoning import *
import os
from GraphReasoning.openai_tools import generate_OpenAIGPT
import PyPDF2
from transformers import AutoTokenizer, AutoModel
import glob
import networkx as nx
import json
import pickle
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


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

    print("Text before processing len: ", len(text))

    extracted_content = generate_openai_llm(
        system_prompt=system_prompt,
        prompt=prompt,
        temperature=0.1,  # Keep temperature low for precise extraction
        max_tokens=10000,  # Use a high max_tokens, as paper content can be long
    )

    if extracted_content.strip() == "NO_CONTENT":
        return ""

    # Some post-processing might be needed if the LLM adds introductory phrases
    # For example, if it says "Here is the extracted content:"
    # You might want to remove such phrases.
    # A common pattern is to just return the LLM's output directly, and trust
    # its instruction to only return the content.

    print("Text after processing len: ", len(extracted_content))

    return extracted_content.strip()


# API Keys
openai_api_key = os.getenv("OPENAI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


class ProgressTracker:
    """Track and save progress for graph generation process."""

    def __init__(self, output_dir="./ambroxol_output_v2/"):
        self.output_dir = output_dir
        self.progress_file = os.path.join(output_dir, "progress.json")
        self.embeddings_file = os.path.join(output_dir, "node_embeddings.pkl")
        self.graph_file = os.path.join(output_dir, "current_graph.graphml")
        self.processed_files = set()
        self.current_graph_path = None
        self.node_embeddings = {}
        self.load_progress()

    def load_progress(self):
        """Load existing progress if available."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, "r") as f:
                    data = json.load(f)
                    self.processed_files = set(data.get("processed_files", []))
                    self.current_graph_path = data.get("current_graph_path")
                    print(
                        f"📋 Loaded progress: {len(self.processed_files)} files already processed"
                    )
                    print(f"📁 Current graph: {self.current_graph_path}")
            except Exception as e:
                print(f"⚠️ Error loading progress: {e}")

        # Load embeddings if available
        if os.path.exists(self.embeddings_file):
            try:
                with open(self.embeddings_file, "rb") as f:
                    self.node_embeddings = pickle.load(f)
                print(f"🔍 Loaded {len(self.node_embeddings)} node embeddings")
            except Exception as e:
                print(f"⚠️ Error loading embeddings: {e}")

    def save_progress(self, processed_files, current_graph_path, node_embeddings=None):
        """Save current progress."""
        self.processed_files = processed_files
        self.current_graph_path = current_graph_path

        # Save progress metadata
        progress_data = {
            "processed_files": list(processed_files),
            "current_graph_path": current_graph_path,
            "last_updated": datetime.now().isoformat(),
            "total_files_processed": len(processed_files),
        }

        try:
            with open(self.progress_file, "w") as f:
                json.dump(progress_data, f, indent=2)
            print(f"💾 Progress saved: {len(processed_files)} files processed")
        except Exception as e:
            print(f"❌ Error saving progress: {e}")

        # Save embeddings if provided
        if node_embeddings is not None:
            try:
                with open(self.embeddings_file, "wb") as f:
                    pickle.dump(node_embeddings, f)
                print(f"💾 Embeddings saved: {len(node_embeddings)} nodes")
            except Exception as e:
                print(f"❌ Error saving embeddings: {e}")

    def is_file_processed(self, file_path):
        """Check if a file has already been processed."""
        return file_path in self.processed_files

    def get_unprocessed_files(self, all_files):
        """Get list of files that haven't been processed yet."""
        return [f for f in all_files if not self.is_file_processed(f)]

    def get_remaining_files(self, all_files):
        """Get count of remaining files to process."""
        return len(self.get_unprocessed_files(all_files))


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
        return process(text)  # Use the process function to clean the text
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


def run(start_from_index=9):  # 0-indexed, so 9 = 10th PDF
    try:
        # Import required functions
        from GraphReasoning.graph_generation import (
            make_graph_from_text,
            add_new_subgraph_from_text,
        )
        from GraphReasoning.graph_tools import generate_node_embeddings

        print(
            f"🚀 Starting knowledge graph generation from PDF {start_from_index + 1}/23..."
        )

        # Initialize progress tracker
        output_dir = "./ambroxol_output_v2/"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        progress_tracker = ProgressTracker(output_dir)

        # Initialize embedding model
        print("📚 Initializing embedding model...")
        tokenizer, model = initialize_embedding_model()

        # Get all PDF files in test-ambroxol directory
        pdf_files = glob.glob("test-ambroxol/*.pdf")
        pdf_files.sort()  # Sort for consistent processing order

        if not pdf_files:
            print("❌ No PDF files found in test-ambroxol directory")
            return

        total_files = len(pdf_files)
        print(f"📄 Found {total_files} total PDF files")

        # Validate start index
        if start_from_index >= total_files:
            print(
                f"❌ Start index {start_from_index} is out of range. Max index is {total_files - 1}"
            )
            return

        # Get files to process (starting from the specified index)
        files_to_process = pdf_files[start_from_index:]
        remaining_count = len(files_to_process)

        print(
            f"📋 Starting from PDF {start_from_index + 1}: {os.path.basename(pdf_files[start_from_index])}"
        )
        print(f"⏳ Files to process: {remaining_count}")

        # Initialize variables
        G = None
        graph_GraphML = progress_tracker.current_graph_path
        node_embeddings = progress_tracker.node_embeddings

        # Mark files before start index as processed (to maintain progress tracking)
        for i in range(start_from_index):
            progress_tracker.processed_files.add(pdf_files[i])

        # Process files starting from the specified index
        for i, pdf_file in enumerate(files_to_process):
            current_file_index = start_from_index + i
            print(
                f"\n📖 Processing PDF {current_file_index + 1}/{total_files}: {os.path.basename(pdf_file)}"
            )

            pdf_text = extract_text_from_pdf(pdf_file)

            if not pdf_text.strip():
                print(f"⚠️ No text extracted from {pdf_file}, skipping...")
                progress_tracker.processed_files.add(pdf_file)
                progress_tracker.save_progress(
                    progress_tracker.processed_files, graph_GraphML, node_embeddings
                )
                continue

            try:
                if G is None:
                    # First file or no existing graph - create initial graph
                    print("🆕 Creating initial graph...")
                    graph_HTML, graph_GraphML, G, net, output_pdf = (
                        make_graph_from_text(
                            txt=pdf_text,
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
                    )

                    # Generate embeddings for the initial graph
                    print("🔍 Generating embeddings for initial graph...")
                    node_embeddings = generate_node_embeddings(G, tokenizer, model)

                    print(f"✅ Initial graph created from {os.path.basename(pdf_file)}")
                    print(
                        f"📊 Initial graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
                    )

                else:
                    # Add to existing graph
                    print("➕ Adding to existing graph...")
                    (
                        graph_GraphML_new,
                        G_new,
                        G_loaded,
                        G_original,
                        node_embeddings,
                        res,
                    ) = add_new_subgraph_from_text(
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

                    # Update the main graph
                    G = G_new
                    graph_GraphML = graph_GraphML_new

                    print(f"✅ Added subgraph from {os.path.basename(pdf_file)}")
                    print(
                        f"📊 Updated graph - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}"
                    )

                # Mark file as processed and save progress
                progress_tracker.processed_files.add(pdf_file)
                progress_tracker.save_progress(
                    progress_tracker.processed_files, graph_GraphML, node_embeddings
                )

                print(
                    f"💾 Progress saved after processing {os.path.basename(pdf_file)}"
                )

            except Exception as e:
                print(f"❌ Error processing {pdf_file}: {e}")
                print("🔄 Process can be resumed from this point later")
                continue

        # Final statistics
        print(f"\n🎉 Knowledge graph generation completed!")
        print(f"📊 Final graph statistics:")
        print(f"   • Nodes: {G.number_of_nodes() if G else 0}")
        print(f"   • Edges: {G.number_of_edges() if G else 0}")
        if G:
            print(f"   • Connected components: {nx.number_connected_components(G)}")
        print(
            f"   • Files processed: {len(progress_tracker.processed_files)}/{total_files}"
        )
        print(f"   • Files saved to: {output_dir}")

        # Save final progress
        progress_tracker.save_progress(
            progress_tracker.processed_files, graph_GraphML, node_embeddings
        )
        print("💾 Final progress saved")

    except ImportError:
        print("❌ GraphReasoning not found. Make sure it's properly installed.")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("🔄 Process can be resumed from the last saved point")


if __name__ == "__main__":
    run(start_from_index=11)
