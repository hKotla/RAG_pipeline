"""CLI entry point: python -m rag.query "<question>" """
import sys


def main(question: str) -> None:
    from rag.config import RAGConfig
    from rag.pipeline import RAGPipeline

    config   = RAGConfig()
    pipeline = RAGPipeline(config)
    pipeline.ask(question, show_sources=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m rag.query "<question>"')
        sys.exit(1)
    main(" ".join(sys.argv[1:]))
