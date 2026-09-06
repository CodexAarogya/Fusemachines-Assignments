"""
Optional: export the embedding model to ONNX for faster CPU inference.

Why the embedding model and not the generative LLM?
ONNX Runtime shines for encoder-only, fixed-shape forward passes -- exactly
what a sentence-transformer embedding call is. The generative LLM instead
uses vLLM (see ARCHITECTURE.md) because autoregressive decoding with a
growing KV-cache is a poor fit for ONNX's static graph export and vLLM's
PagedAttention gives materially better throughput for that workload.

Usage:
    pip install optimum[onnxruntime]
    python scripts/export_embedding_onnx.py --model sentence-transformers/all-MiniLM-L6-v2 --out ./onnx_embedding_model
"""
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out", default="./onnx_embedding_model")
    args = parser.parse_args()

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    print(f"Exporting {args.model} to ONNX at {args.out} ...")
    model = ORTModelForFeatureExtraction.from_pretrained(args.model, export=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("Done. Point EmbeddingModel at this ONNX model to serve with onnxruntime "
          "instead of PyTorch for lower CPU latency.")


if __name__ == "__main__":
    main()
