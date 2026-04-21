import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SmolVLM2 model files from Hugging Face.")
    parser.add_argument(
        "--repo",
        default="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        help="Hugging Face repo id",
    )
    parser.add_argument(
        "--output",
        default="download/smolvlm2-model",
        help="Output directory for model files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=args.repo,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"SmolVLM2 model downloaded to {output_dir}")


if __name__ == "__main__":
    main()
