from src.pipeline import index_documents, retrieve_relevant_chunks
from src.answer_rest import generate_answer
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index",
        action="store_true",
    )
    parser.add_argument(
        "--question",
        type=str,
    )

    args = parser.parse_args()

    if args.index:
        index_documents()
        return

    if not args.question:
        parser.error("You must provide a question using --question")

    chunks = retrieve_relevant_chunks(args.question, top_k=5)
    answer = generate_answer(args.question, chunks)

    print("\n=== QUESTION ===\n")
    print(args.question)
    print("\n=== ANSWER ===\n")
    print(answer)


if __name__ == "__main__":
    main()
