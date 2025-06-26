"""
Build ES (Elasticsearch) BM25 Index.
"""

from typing import Dict, Optional
import json
import argparse
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from typing import Any
import hashlib
import io
import dill
from tqdm import tqdm
import glob
import bz2
import base58
import os


def hash_object(o: Any) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def make_hotpotqa_documents(elasticsearch_index: str, metadata: Optional[Dict] = None):
    raw_glob_filepath = os.path.join("raw_data", "hotpotqa", "wikpedia-paragraphs", "*", "wiki_*.bz2")
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata
    for filepath in tqdm(glob.glob(raw_glob_filepath)):
        for datum in bz2.BZ2File(filepath).readlines():
            instance = json.loads(datum.strip())

            id_ = hash_object(instance)[:32]
            title = instance["title"]
            sentences_text = [e.strip() for e in instance["text"]]
            paragraph_text = " ".join(sentences_text)
            url = instance["url"]
            is_abstract = True
            paragraph_index = 0

            es_paragraph = {
                "id": id_,
                "title": title,
                "paragraph_index": paragraph_index,
                "paragraph_text": paragraph_text,
                "url": url,
                "is_abstract": is_abstract,
            }
            document = {
                "_op_type": "create",
                "_index": elasticsearch_index,
                "_id": metadata["idx"],
                "_source": es_paragraph,
            }
            yield (document)
            metadata["idx"] += 1


def make_2wikimultihopqa_documents(elasticsearch_index: str, metadata: Optional[Dict] = None):
    raw_filepaths = [
        os.path.join("raw_data", "2wikimultihopqa", "train.json"),
        os.path.join("raw_data", "2wikimultihopqa", "dev.json"),
        os.path.join("raw_data", "2wikimultihopqa", "test.json"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            full_data = json.load(file)
            for instance in tqdm(full_data):
                for paragraph in instance["context"]:
                    title = paragraph[0]
                    paragraph_text = " ".join(paragraph[1])
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                    }
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1


def make_musique_documents(elasticsearch_index: str, metadata: Optional[Dict] = None):
    raw_filepaths = [
        os.path.join("raw_data", "musique", "musique_ans_v1.0_dev.jsonl"),
        os.path.join("raw_data", "musique", "musique_ans_v1.0_test.jsonl"),
        os.path.join("raw_data", "musique", "musique_ans_v1.0_train.jsonl"),
        os.path.join("raw_data", "musique", "musique_full_v1.0_dev.jsonl"),
        os.path.join("raw_data", "musique", "musique_full_v1.0_test.jsonl"),
        os.path.join("raw_data", "musique", "musique_full_v1.0_train.jsonl"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:
        with open(raw_filepath, "r") as file:
            for line in tqdm(file.readlines()):
                if not line.strip():
                    continue
                instance = json.loads(line)

                for paragraph in instance["paragraphs"]:
                    title = paragraph["title"]
                    paragraph_text = paragraph["paragraph_text"]
                    paragraph_index = 0
                    url = ""
                    is_abstract = paragraph_index == 0

                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": paragraph_index,
                        "paragraph_text": paragraph_text,
                        "url": url,
                        "is_abstract": is_abstract,
                    }
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1


def make_crag_documents(elasticsearch_index: str, metadata: Optional[Dict] = None):
    """
    Generates Elasticsearch documents from the raw CRAG dataset.
    This function reads the original CRAG data, extracts all unique web pages
    from the 'search_results' of all questions, and prepares them for indexing.
    """
    # This should point to your ORIGINAL, UNPROCESSED CRAG data file.
    # Please adjust the filenames as necessary.
    raw_filepaths = [
        os.path.join("raw_data", "crag", "train.jsonl"),
        os.path.join("raw_data", "crag", "dev.jsonl"),
        os.path.join("raw_data", "crag", "test.jsonl"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    used_full_ids = set()
    for raw_filepath in raw_filepaths:
        if not os.path.exists(raw_filepath):
            print(f"Warning: Raw data file not found, skipping: {raw_filepath}")
            continue

        with open(raw_filepath, "r") as file:
            for line in tqdm(file.readlines(), desc=f"Processing raw CRAG file: {raw_filepath}"):
                if not line.strip():
                    continue
                # We are reading the raw CRAG data here
                instance = json.loads(line)

                # Iterate through the search results which form our corpus
                for search_result in instance.get("search_results", []):
                    title = search_result.get("page_name", "No Title").strip()

                    # Here you might want to parse HTML to get clean text
                    # For now, we'll use the raw HTML as paragraph_text
                    paragraph_text = search_result.get("page_result", "").strip()

                    # Create a unique ID for the document to avoid duplicates
                    full_id = hash_object(" ".join([title, paragraph_text]))
                    if full_id in used_full_ids:
                        continue

                    used_full_ids.add(full_id)
                    id_ = full_id[:32]

                    es_paragraph = {
                        "id": id_,
                        "title": title,
                        "paragraph_index": 0,  # Index is 0 as we treat each page as one doc
                        "paragraph_text": paragraph_text,
                        "url": search_result.get("page_url", ""),
                        "is_abstract": True,  # Assuming each page is a primary document
                    }
                    document = {
                        "_op_type": "create",
                        "_index": elasticsearch_index,
                        "_id": metadata["idx"],
                        "_source": es_paragraph,
                    }
                    yield (document)
                    metadata["idx"] += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index paragraphs in Elasticsearch")
    parser.add_argument(
        "dataset_name",
        help="name of the dataset",
        type=str,
        choices=("hotpotqa", "2wikimultihopqa", "musique", "crag"),
    )
    parser.add_argument(
        "--force",
        help="force delete before creating new index.",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    # conntect elastic-search
    elastic_host = "localhost"
    elastic_port = 9200
    elasticsearch_index = args.dataset_name
    es = Elasticsearch(
        [{"host": elastic_host, "port": elastic_port}],
        max_retries=2,  # it's exp backoff starting 2, more than 2 retries will be too much.
        timeout=500,
        retry_on_timeout=True,
    )

    paragraphs_index_settings = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "english",
                },
                "paragraph_index": {"type": "integer"},
                "paragraph_text": {
                    "type": "text",
                    "analyzer": "english",
                },
                "url": {
                    "type": "text",
                    "analyzer": "english",
                },
                "is_abstract": {"type": "boolean"},
            }
        }
    }

    index_exists = es.indices.exists(elasticsearch_index)
    print("Index already exists" if index_exists else "Index doesn't exist.")

    # delete index if exists
    if index_exists:
        if not args.force:
            feedback = input(f"Index {elasticsearch_index} already exists. " f"Are you sure you want to delete it?")
            if not (feedback.startswith("y") or feedback == ""):
                exit("Termited by user.")
        es.indices.delete(index=elasticsearch_index)

    # create index
    print("Creating Index ...")
    es.indices.create(index=elasticsearch_index, body=paragraphs_index_settings)

    if args.dataset_name == "hotpotqa":
        make_documents = make_hotpotqa_documents
    elif args.dataset_name == "2wikimultihopqa":
        make_documents = make_2wikimultihopqa_documents
    elif args.dataset_name == "musique":
        make_documents = make_musique_documents
    elif args.dataset_name == "crag":
        make_documents = make_crag_documents
    else:
        raise Exception(f"Unknown dataset_name {args.dataset_name}")

    # Bulk-insert documents into index
    print("Inserting Paragraphs ...")
    result = bulk(
        es,
        make_documents(elasticsearch_index),
        raise_on_error=True,  # set to true o/w it'll fail silently and only show less docs.
        raise_on_exception=True,  # set to true o/w it'll fail silently and only show less docs.
        max_retries=2,  # it's exp backoff starting 2, more than 2 retries will be too much.
        request_timeout=500,
    )
    es.indices.refresh(elasticsearch_index)  # actually updates the count.
    document_count = result[0]
    print(f"Index {elasticsearch_index} is ready. Added {document_count} documents.")
