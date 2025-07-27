import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType


def convert_sentence_transformer_to_onnx(
    model_name: str, 
    output_dir: str, 
    quantize: bool = True,
    opset_version: int = 13
) -> Dict[str, str]:
    """
    Convert a SentenceTransformer model to ONNX format.
    
    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save the ONNX model
        quantize: Whether to quantize the model
        opset_version: ONNX opset version
        
    Returns:
        Dictionary with paths to the ONNX model files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get model name for file naming
    model_filename = model_name.split("/")[-1]
    onnx_path = output_dir / f"{model_filename}.onnx"
    quantized_path = output_dir / f"{model_filename}_quantized.onnx"
    
    # Load model
    print(f"Loading model {model_name}...")
    model = SentenceTransformer(model_name)
    
    # Prepare dummy input
    dummy_input = torch.ones(1, 128, dtype=torch.int64)  # (batch_size, seq_length)
    attention_mask = torch.ones(1, 128, dtype=torch.int64)
    token_type_ids = torch.zeros(1, 128, dtype=torch.int64)
    
    # Export to ONNX
    print(f"Converting model to ONNX format...")
    with torch.no_grad():
        symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
        torch.onnx.export(
            model,
            (dummy_input, attention_mask, token_type_ids),
            onnx_path,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['sentence_embedding'],
            dynamic_axes={
                'input_ids': symbolic_names,
                'attention_mask': symbolic_names,
                'token_type_ids': symbolic_names,
                'sentence_embedding': {0: 'batch_size'}
            }
        )
    
    # Verify the ONNX model
    print("Verifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    
    # Quantize the model if requested
    if quantize:
        print("Quantizing ONNX model...")
        quantize_dynamic(
            str(onnx_path),
            str(quantized_path),
            weight_type=QuantType.QInt8
        )
        model_path = quantized_path
    else:
        model_path = onnx_path
    
    print(f"Model conversion completed. Model saved to {model_path}")
    
    return {
        "original_onnx_path": str(onnx_path),
        "quantized_onnx_path": str(quantized_path) if quantize else None,
        "final_model_path": str(model_path)
    }


def convert_cross_encoder_to_onnx(
    model_name: str, 
    output_dir: str, 
    quantize: bool = True,
    opset_version: int = 13
) -> Dict[str, str]:
    """
    Convert a Cross-Encoder model to ONNX format.
    
    Args:
        model_name: HuggingFace model name or path
        output_dir: Directory to save the ONNX model
        quantize: Whether to quantize the model
        opset_version: ONNX opset version
        
    Returns:
        Dictionary with paths to the ONNX model files
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Get model name for file naming
    model_filename = model_name.split("/")[-1]
    onnx_path = output_dir / f"{model_filename}.onnx"
    quantized_path = output_dir / f"{model_filename}_quantized.onnx"
    
    # Load model
    print(f"Loading model {model_name}...")
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
        model = CrossEncoder(model_name)
        
        # Get the underlying PyTorch model
        torch_model = model.model
        
        # Prepare dummy input
        dummy_input = torch.ones(1, 128, dtype=torch.int64)  # (batch_size, seq_length)
        attention_mask = torch.ones(1, 128, dtype=torch.int64)
        
        # Export to ONNX
        print(f"Converting model to ONNX format...")
        with torch.no_grad():
            symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
            torch.onnx.export(
                torch_model,
                (dummy_input, attention_mask),
                onnx_path,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input_ids', 'attention_mask'],
                output_names=['score'],
                dynamic_axes={
                    'input_ids': symbolic_names,
                    'attention_mask': symbolic_names,
                    'score': {0: 'batch_size'}
                }
            )
        
        # Verify the ONNX model
        print("Verifying ONNX model...")
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Quantize the model if requested
        if quantize:
            print("Quantizing ONNX model...")
            quantize_dynamic(
                str(onnx_path),
                str(quantized_path),
                weight_type=QuantType.QInt8
            )
            model_path = quantized_path
        else:
            model_path = onnx_path
        
        print(f"Model conversion completed. Model saved to {model_path}")
        
        return {
            "original_onnx_path": str(onnx_path),
            "quantized_onnx_path": str(quantized_path) if quantize else None,
            "final_model_path": str(model_path)
        }
    except ImportError:
        print("CrossEncoder not available, skipping conversion")
        return {
            "original_onnx_path": None,
            "quantized_onnx_path": None,
            "final_model_path": None
        }


def convert_models(output_dir: str = "models"):
    """
    Convert and quantize both retrieval and reranking models.
    
    Args:
        output_dir: Directory to save the models
    """
    # Convert retrieval model
    retrieval_model = "sentence-transformers/all-MiniLM-L6-v2"
    retrieval_output = convert_sentence_transformer_to_onnx(retrieval_model, output_dir)
    
    # Convert reranking model
    reranking_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranking_output = convert_cross_encoder_to_onnx(reranking_model, output_dir)
    
    # Save model paths
    model_paths = {
        "retrieval_model": retrieval_output,
        "reranking_model": reranking_output
    }
    
    with open(os.path.join(output_dir, "model_paths.json"), "w") as f:
        import json
        json.dump(model_paths, f, indent=4)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert and quantize models to ONNX format")
    parser.add_argument(
        "--output-dir", "-o",
        default="models",
        help="Directory to save the models (default: models)"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization step"
    )
    
    args = parser.parse_args()
    
    # Convert retrieval model
    retrieval_model = "sentence-transformers/all-MiniLM-L6-v2"
    convert_sentence_transformer_to_onnx(
        retrieval_model, 
        args.output_dir,
        quantize=not args.no_quantize
    )
    
    # Convert reranking model
    reranking_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
    convert_cross_encoder_to_onnx(
        reranking_model, 
        args.output_dir,
        quantize=not args.no_quantize
    ) 