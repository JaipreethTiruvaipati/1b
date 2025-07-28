#!/usr/bin/env python3
"""
Profiler for benchmarking the document analysis system.
This script runs a series of tests to measure performance metrics.
"""

import os
import time
import json
import argparse
import cProfile
import pstats
import io
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Add the parent directory to the path to allow importing the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_analyzer import DocumentAnalyzer
from src.test_system import create_mock_sections, test_retrieval_system


def profile_func(func, *args, **kwargs):
    """
    Profile a function and return stats.
    
    Args:
        func: Function to profile
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Tuple of (result, stats)
    """
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start_time
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Print top 20 functions
    
    return result, elapsed, s.getvalue()


def run_benchmark(num_documents=5, num_iterations=3):
    """
    Run a benchmark with varying document counts.
    
    Args:
        num_documents: Number of documents to process
        num_iterations: Number of iterations to run for each configuration
    """
    print(f"Running benchmark with {num_documents} documents, {num_iterations} iterations...")
    
    # Create a mock analyzer
    analyzer = DocumentAnalyzer("data", "output")
    
    # Create sections for the mock documents
    sections = create_mock_sections()
    
    # Track metrics
    metrics = {
        "section_extraction": [],
        "indexing": [],
        "search": [],
        "total": []
    }
    
    # Run benchmark
    for i in range(num_iterations):
        print(f"Iteration {i+1}/{num_iterations}")
        
        # Time section extraction
        start = time.time()
        # Mock the section extraction since we're using fake sections
        time.sleep(0.1)  # Simulate some work
        extraction_time = time.time() - start
        metrics["section_extraction"].append(extraction_time)
        
        # Create retrieval system and time indexing
        from src.retrieval_system import RetrievalSystem
        retrieval_system = RetrievalSystem()
        
        start = time.time()
        retrieval_system.index_sections(sections)
        indexing_time = time.time() - start
        metrics["indexing"].append(indexing_time)
        
        # Time search
        query = "Plan a trip of 4 days for a group of 10 college friends"
        start = time.time()
        results = retrieval_system.search(query, k=5)
        search_time = time.time() - start
        metrics["search"].append(search_time)
        
        # Calculate total time
        total_time = extraction_time + indexing_time + search_time
        metrics["total"].append(total_time)
        
        print(f"  Section extraction: {extraction_time:.4f}s")
        print(f"  Indexing: {indexing_time:.4f}s")
        print(f"  Search: {search_time:.4f}s")
        print(f"  Total: {total_time:.4f}s")
    
    # Calculate averages
    averages = {
        key: sum(values) / len(values) for key, values in metrics.items()
    }
    
    print("\nAverage times:")
    print(f"  Section extraction: {averages['section_extraction']:.4f}s")
    print(f"  Indexing: {averages['indexing']:.4f}s")
    print(f"  Search: {averages['search']:.4f}s")
    print(f"  Total: {averages['total']:.4f}s")
    
    return metrics, averages


def generate_performance_report(metrics, averages, output_dir="output"):
    """
    Generate a performance report with charts.
    
    Args:
        metrics: Dictionary of metrics
        averages: Dictionary of average metrics
        output_dir: Directory to save the report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = list(averages.keys())
    values = [averages[key] for key in labels]
    
    ax.bar(labels, values)
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Average Performance Metrics')
    
    # Save chart
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    chart_path = output_dir / f"performance_chart_{timestamp}.png"
    plt.savefig(chart_path)
    
    # Save metrics as JSON
    metrics_path = output_dir / f"performance_metrics_{timestamp}.json"
    
    with open(metrics_path, "w") as f:
        json.dump({
            "metrics": metrics,
            "averages": averages,
            "timestamp": timestamp
        }, f, indent=2)
    
    print(f"Performance report saved to {output_dir}")
    print(f"  Chart: {chart_path}")
    print(f"  Metrics: {metrics_path}")


def profile_analysis():
    """Profile the full document analysis process."""
    # Create mock input data
    input_data = {
        "challenge_info": {
            "challenge_id": "test_001",
            "test_case_name": "test",
            "description": "Test"
        },
        "documents": [
            {"filename": "doc1.pdf", "title": "Document 1"},
            {"filename": "doc2.pdf", "title": "Document 2"},
            {"filename": "doc3.pdf", "title": "Document 3"}
        ],
        "persona": {
            "role": "Test Persona"
        },
        "job_to_be_done": {
            "task": "Test task"
        }
    }
    
    # Mock the analyzer to use our test sections
    class MockAnalyzer(DocumentAnalyzer):
        def __init__(self, data_dir, output_dir):
            super().__init__(data_dir, output_dir)
        
        def _extract_sections(self, doc_files):
            # Return mock sections instead of extracting from files
            return create_mock_sections()
    
    # Create analyzer
    analyzer = MockAnalyzer("data", "output")
    
    # Profile the analyze method
    print("Profiling document analysis...")
    result, elapsed, stats = profile_func(analyzer.analyze, input_data)
    
    print(f"\nAnalysis completed in {elapsed:.4f} seconds")
    print("\nProfile stats (top 20 functions):")
    print(stats)
    
    return result, elapsed, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile the document analysis system")
    parser.add_argument(
        "--documents", "-d",
        type=int,
        default=5,
        help="Number of documents to process"
    )
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=3,
        help="Number of iterations for each benchmark"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory to save the performance report"
    )
    parser.add_argument(
        "--full-profile",
        action="store_true",
        help="Run a full profile of the analysis process"
    )
    
    args = parser.parse_args()
    
    if args.full_profile:
        # Profile the full analysis process
        profile_analysis()
    else:
        # Run the benchmark
        metrics, averages = run_benchmark(args.documents, args.iterations)
        generate_performance_report(metrics, averages, args.output_dir) 