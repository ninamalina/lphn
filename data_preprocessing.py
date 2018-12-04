import warnings
warnings.filterwarnings("ignore")
import json
import time
import io


def get_object_index(object_index_dict, object_id):
    if object_id not in object_index_dict:
        object_index_dict[object_id] = len(object_index_dict)
    return object_index_dict[object_id]

def preprocess_dlbp_data(in_files):

    citations = open("data/dblp/parsed/paper_paper.tsv", "w+")
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
                paper_id = data["id"]
                paper_index = get_object_index(paper_index_dict, paper_id)

                if "title" in data:
                    paper_title_dict[paper_id] = data["title"]

                if "references" in data:
                    references = data["references"]
                    for reference in references:
                        reference_index = get_object_index(paper_index_dict, reference)
                        citations.write(str(paper_index) + "\t" + str(reference_index) + "\n")
                
                if "authors" in data:
                    authors = data["authors"]
                    for author in authors:
                        author_index = get_object_index(author_index_dict, author)
                        author_paper.write(str(author_index) + "\t" + str(paper_index) + "\n")

                if "venue" in data:
                    venue = data["venue"]
                    if venue != "":
                        venue_index = get_object_index(venue_index_dict, venue)
                    
                        paper_venue.write(str(paper_index) + "\t" + str(venue_index) + "\n")
                
                    
                i += 1
                if i % 100000 == 0:
                    print(i, time.time() - times_start)
                    times_start = time.time()

    with io.open("data/dblp/parsed/index_author.tsv", "w+", encoding="utf-8") as f:
        for author in author_index_dict:
            f.write(str(author_index_dict[author]) + "\t" + author + "\n")

    with io.open("data/dblp/parsed/index_paper.tsv", "w+") as f:
        for paper in paper_index_dict:
            f.write(str(paper_index_dict[paper]) + "\t" + paper + "\n")

    with io.open("data/dblp/parsed/index_venue.tsv", "w+", encoding="utf-8") as f:
        for venue in venue_index_dict:
            f.write(str(venue_index_dict[venue]) + "\t" + venue + "\n")

    with io.open("data/dblp/parsed/paper_title.tsv", "w+", encoding="utf-8") as f:
        for paper_id in paper_title_dict:
            f.write(paper_id + "\t" + paper_title_dict[paper_id] + "\n")


def preprocess_bio_data(gene_gene, disease_gene, drug_gene):
    
    gene_index_dict = {}
    disease_index_dict = {}
    drug_index_dict = {}
    
    with open(gene_gene) as f_in:    
        with open("data/bio/parsed/gene_gene.tsv", "w+") as f_out:
            for line in f_in:
                splited = line.split(",")
                gene1 = splited[0].strip()
                gene2 = splited[1].strip()
                gene1_index = get_object_index(gene_index_dict, gene1)
                gene2_index = get_object_index(gene_index_dict, gene2)
                f_out.write(str(gene1_index) + "\t" + str(gene2_index) + "\n")
                
    with open(disease_gene) as f_in:    
        f_in.readline()
        with open("data/bio/parsed/disease_gene.tsv", "w+") as f_out:
            for line in f_in:
                splited = line.split("\t")
                disease = splited[0].strip()
                gene = splited[2].strip()
                disease_index = get_object_index(disease_index_dict, disease)
                gene_index = get_object_index(gene_index_dict, gene)
                f_out.write(str(disease_index) + "\t" + str(gene_index) + "\n")

    with open(drug_gene) as f_in:    
        f_in.readline()
        with open("data/bio/parsed/drug_gene.tsv", "w+") as f_out:
            for line in f_in:
                splited = line.split(",")
                drug = splited[0].strip()
                gene = splited[1].strip()
                drug_index = get_object_index(drug_index_dict, drug)
                gene_index = get_object_index(gene_index_dict, gene)
                f_out.write(str(drug_index) + "\t" + str(gene_index) + "\n")

    with open("data/bio/parsed/protein_index.tsv", "w+") as f:
        for gene_id in gene_index_dict:
            f.write(gene_id + "\t" + str(gene_index_dict[gene_id]) + "\n")

    with open("data/bio/parsed/disease_index.tsv", "w+") as f:
        for disease_id in disease_index_dict:
            f.write(disease_id + "\t" + str(disease_index_dict[disease_id]) + "\n")

    with open("data/bio/parsed/drug_index.tsv", "w+") as f:
        for drug_id in drug_index_dict:
            f.write(drug_id + "\t" + str(drug_index_dict[drug_id]) + "\n")
        

def build_edgelist(in_files, dataset):
    
    with open(dataset + "_edgelist.tsv", "w+") as out_file:
        for f_name in in_files:
            first = f_name.split("/")[-1].split(".")[0].split("_")[0]
            second = f_name.split("/")[-1].split(".")[0].split("_")[1]
            with open(f_name) as f:
                for line in f:
                    splited = line.strip().split("\t")
                    out_file.write(first + "_" + splited[0] + "\t" + second + "_" + splited[1] + "\n")
                    


if __name__ == '__main__':

    # preprocess dblp data
    preprocess_dlbp_data(["data/dblp/dblp-ref-0.json", "data/dblp/dblp-ref-1.json", "data/dblp/dblp-ref-2.json", "data/dblp/dblp-ref-3.json"])
    build_edgelist(["data/dblp/parsed/author_paper.tsv", "data/dblp/parsed/paper_paper.tsv", "data/dblp/parsed/paper_venue.tsv"], dataset="data/dblp/parsed/dblp")

    # preprocess bio data
    # preprocess_bio_data(gene_gene="data/bio/PP-Decagon_ppi.csv", disease_gene="data/bio/DG-AssocMiner_miner-disease-gene.tsv", drug_gene="data/bio/ChG-TargetDecagon_targets.csv")
    # build_edgelist(["data/bio/parsed/drug_gene.tsv", "data/bio/parsed/disease_gene.tsv", "data/bio/parsed/gene_gene.tsv"], dataset="data/bio/parsed/bio")
    
    pass
