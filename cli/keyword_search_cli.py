#!/usr/bin/env python3

import argparse
from ast import arg
from lib.keyword_search import search_command,build_command,tf_command,idf_command,tfidf_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Build BM25 index")

    search_parser = subparsers.add_parser("tf", help="Build term frequency")
    search_parser.add_argument("doc_id", type=int, help="doc id to check")
    search_parser.add_argument("term", type=str, help="term to which we need to find count for")

    search_parser = subparsers.add_parser("idf", help="Build inverse document frequency")
    search_parser.add_argument("term", type=str, help="term to which we need to find idf for")

    search_parser = subparsers.add_parser("tfidf", help="Build term frequency Inverse Document frequency")
    search_parser.add_argument("doc_id", type=int, help="doc id to check")
    search_parser.add_argument("term", type=str, help="term to which we need to find count for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"searching for {args.query}")
            results = search_command(args.query)
            
            for i,result in enumerate(results):
                print(f"{i}.{result['title']}")

        case 'build':
            build_command()

        case 'tf':
            tf_command(args.doc_id,args.term)

        case 'idf':
            idf_command(args.term)

        case 'tfidf':
            tfidf_command(args.doc_id,args.term)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()