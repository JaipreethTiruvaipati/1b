#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
import argparse
import sys

# Add the parent directory to the path to allow importing the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # When running as a module
    from src.document_processor import DocumentProcessor, Section
    from src.retrieval_system import Retriever, Reranker, RetrievalSystem
except ImportError:
    # When running directly
    from document_processor import DocumentProcessor, Section
    from retrieval_system import Retriever, Reranker, RetrievalSystem


def create_mock_sections() -> list:
    """Create mock sections for testing without PDF processing."""
    sections = [
        Section(
            document="South of France - Cities.pdf",
            section_title="Comprehensive Guide to Major Cities in the South of France",
            text="""
            The South of France is home to some of the most beautiful cities in Europe.
            Nice is known for its beautiful promenade and beaches.
            Marseille is France's oldest city and second-largest city after Paris.
            Cannes is famous for its international film festival.
            Aix-en-Provence is known for its elegant boulevards and fountains.
            Saint-Tropez is a coastal town known for beaches and nightlife.
            """,
            page_number=1
        ),
        Section(
            document="South of France - Cuisine.pdf",
            section_title="Culinary Experiences",
            text="""
            In addition to dining at top restaurants, there are several culinary experiences you should consider:
            Cooking Classes - Many towns and cities in the South of France offer cooking classes where you can learn to prepare traditional dishes like bouillabaisse, ratatouille, and tarte tropézienne.
            These classes are a great way to immerse yourself in the local culture and gain hands-on experience with regional recipes.
            Some classes even include a visit to a local market to shop for fresh ingredients.
            Wine Tours - The South of France is renowned for its wine regions, including Provence and Languedoc.
            Take a wine tour to visit vineyards, taste local wines, and learn about the winemaking process.
            Many wineries offer guided tours and tastings, giving you the opportunity to sample a variety of wines and discover new favorites.
            """,
            page_number=6
        ),
        Section(
            document="South of France - Things to Do.pdf",
            section_title="Coastal Adventures",
            text="""
            The South of France is renowned for its beautiful coastline along the Mediterranean Sea.
            Here are some activities to enjoy by the sea: Beach Hopping:
            Nice - Visit the sandy shores and enjoy the vibrant Promenade des Anglais;
            Antibes - Relax on the pebbled beaches and explore the charming old town;
            Saint-Tropez - Experience the exclusive beach clubs and glamorous atmosphere;
            Marseille to Cassis - Explore the stunning limestone cliffs and hidden coves of Calanques National Park;
            Îles d'Hyères - Discover pristine beaches and excellent snorkeling opportunities on islands like Porquerolles and Port-Cros;
            Cannes - Enjoy the sandy beaches and luxury beach clubs along the Boulevard de la Croisette;
            Menton - Visit the serene beaches and beautiful gardens in this charming town near the Italian border.
            """,
            page_number=2
        ),
        Section(
            document="South of France - Things to Do.pdf",
            section_title="Nightlife and Entertainment",
            text="""
            The South of France offers a vibrant nightlife scene, with options ranging from chic bars to lively nightclubs:
            Bars and Lounges - Monaco: Enjoy classic cocktails and live jazz at Le Bar Americain, located in the Hôtel de Paris;
            Nice: Try creative cocktails at Le Comptoir du Marché, a trendy bar in the old town;
            Cannes: Experience dining and entertainment at La Folie Douce, with live music, DJs, and performances;
            Marseille: Visit Le Trolleybus, a popular bar with multiple rooms and music styles;
            Saint-Tropez: Relax at Bar du Port, known for its chic atmosphere and waterfront views.
            Nightclubs - Saint-Tropez: Dance at the famous Les Caves du Roy, known for its glamorous atmosphere and celebrity clientele;
            Nice: Party at High Club on the Promenade des Anglais, featuring multiple dance floors and top DJs;
            Cannes: Enjoy the stylish setting and rooftop terrace at La Suite, offering stunning views of Cannes.
            """,
            page_number=11
        ),
        Section(
            document="South of France - Tips and Tricks.pdf",
            section_title="General Packing Tips and Tricks",
            text="""
            General Packing Tips and Tricks: Layering - The weather can vary, so pack layers to stay comfortable in different temperatures;
            Versatile Clothing - Choose items that can be mixed and matched to create multiple outfits, helping you pack lighter;
            Packing Cubes - Use packing cubes to organize your clothes and maximize suitcase space;
            Roll Your Clothes - Rolling clothes saves space and reduces wrinkles;
            Travel-Sized Toiletries - Bring travel-sized toiletries to save space and comply with airline regulations;
            Reusable Bags - Pack a few reusable bags for laundry, shoes, or shopping;
            First Aid Kit - Include a small first aid kit with band-aids, antiseptic wipes, and any necessary medications;
            Copies of Important Documents - Make copies of your passport, travel insurance, and other important documents. Keep them separate from the originals.
            """,
            page_number=2
        )
    ]
    return sections


def test_retrieval_system(query: str, sections: list):
    """Test the retrieval system with mock sections."""
    print(f"Testing retrieval system with query: '{query}'")
    
    # Create retrieval system
    retrieval_system = RetrievalSystem()
    
    # Index sections
    retrieval_system.index_sections(sections)
    
    # Search
    print("\nSearching...")
    results = retrieval_system.search(query, k=5)
    
    # Print results
    print("\nSearch results:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Document: {result['document']}")
        print(f"   Section: {result['section_title']}")
        print(f"   Page: {result['page_number']}")
        print(f"   Score: {result['score']:.4f}\n")
    
    return results


def test_mock_analysis():
    """
    Test the system with mock sections instead of real PDF processing.
    This allows testing the retrieval and ranking functionality without real PDFs.
    """
    # Create mock sections
    sections = create_mock_sections()
    
    # Test queries
    queries = [
        "Plan a beach vacation for college friends",
        "Find nightlife options for young travelers",
        "Recommend culinary experiences for a group",
        "What to pack for a trip to the South of France"
    ]
    
    # Test each query
    for query in queries:
        print("=" * 80)
        test_retrieval_system(query, sections)
        print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the document analysis system")
    parser.add_argument(
        "--query", "-q",
        default="Plan a trip of 4 days for a group of 10 college friends",
        help="Query to test with"
    )
    
    args = parser.parse_args()
    
    # Create mock sections
    sections = create_mock_sections()
    
    # Test with the provided query
    test_retrieval_system(args.query, sections) 