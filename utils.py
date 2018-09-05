import warnings
warnings.filterwarnings("ignore")
import networkx as nx 
import json
from pprint import pprint
import time
import io


def get_object_index(object_index_dict, object_id):
    if object_id not in object_index_dict:
        object_index_dict[object_id] = len(object_index_dict)
    return object_index_dict[object_id]


def preprocess_dlbp_data(in_files):

    citations = open("data/dblp/parsed/paper_reference.tsv", "w+")
    paper_venue = open("data/dblp/parsed/paper_venue.tsv", "w+")
    author_paper = open("data/dblp/parsed/author_paper.tsv", "w+")

    paper_index_dict = {}
    author_index_dict = {}
    venue_index_dict = {}
    paper_title_dict = {}

    i = 0

    for f_name in in_files:
        with open(f_name) as f:
            
            times_start = time.time()

            for line in f:
                data = json.loads(line) # json corresponding to one paper
                # print(data)
                paper_id = data["id"]
                paper_index = get_object_index(paper_index_dict, paper_id)

                if "title" in data:
                    paper_title_dict[paper_id] = data["title"]

                if "references" in data:
                    references = data["references"]
                    for reference_id in references:
                        reference_id = reference_id
                        reference_index = get_object_index(paper_index_dict, reference_id)
                        citations.write(str(paper_index) + "\t" + str(reference_index) + "\n")
                
                if "authors" in data:
                    authors = data["authors"]
                    for author in authors:
                        author = author
                        author_index = get_object_index(author_index_dict, author)
                        author_paper.write(str(author_index) + "\t" + str(paper_index) + "\n")

                if "venue" in data:
                    venue = data["venue"]
                    venue_index = get_object_index(venue_index_dict, venue)
                    paper_venue.write(str(paper_index) + "\t" + str(venue_index) + "\n")
                
                i += 1
                if i % 100000 == 0:
                    print(i, time.time() - times_start)
                    times_start = time.time()
                    # break

    with io.open("author_index.tsv", "w+", encoding="utf-8") as f:
        for author_id in author_index_dict:
            f.write(author_id + "\t" + str(author_index_dict[author_id]) + "\n")

    with io.open("paper_index.tsv", "w+") as f:
        for paper_id in paper_index_dict:
            f.write(paper_id + "\t" + str(paper_index_dict[paper_id]) + "\n")

    with io.open("venue_index.tsv", "w+", encoding="utf-8") as f:
        for venue_id in venue_index_dict:
            f.write(venue_id + "\t" + str(venue_index_dict[venue_id]) + "\n")

    with io.open("paper_title.tsv", "w+", encoding="utf-8") as f:
        for paper_id in paper_title_dict:
            # print(paper_title_dict[paper_id].decode('ascii', 'ignore'), type(paper_title_dict[paper_id].decode('ascii', 'ignore')) )
            # print(paper_title_dict[paper_id], type(paper_title_dict[paper_id]))
            # print(paper_title_dict[paper_id].encode("utf-8"), type(paper_title_dict[paper_id].encode("utf-8")))
            # print(paper_title_dict[paper_id].encode("utf-8"), type(paper_title_dict[paper_id].encode("utf-8")))
            # print(paper_title_dict[paper_id].encode("utf-8")[83])
            f.write(paper_id + "\t" + paper_title_dict[paper_id] + "\n")



if __name__ == '__main__':
    preprocess_dlbp_data(["data/dblp/dblp-ref-0.json", "data/dblp/dblp-ref-1.json", "data/dblp/dblp-ref-2.json", "data/dblp/dblp-ref-3.json"])
