import sys
import os
import random
from collections import Counter
from collections import defaultdict

class MetaPathGeneratorSicris:
    def __init__(self, seed):
        self.field_authors = defaultdict(list)
        self.author_fields = defaultdict(list)
        self.paper_authors = defaultdict(list)
        self.author_papers = defaultdict(list)
        random.seed(seed)

    def read_data(self, G):
        for edge in G.edges():
            first_type = edge[0].split("_")[0]
            second_type = edge[1].split("_")[0]
            if first_type == "author" and second_type == "field":
                self.author_fields[edge[0]].append(edge[1])
                self.field_authors[edge[1]].append(edge[0])
            elif first_type == "field" and second_type == "author":
                self.author_fields[edge[1]].append(edge[0])
                self.field_authors[edge[0]].append(edge[1])
            elif first_type == "author" and second_type == "paper":
                self.paper_authors[edge[1]].append(edge[0])
                self.author_papers[edge[0]].append(edge[1])
            elif first_type == "paper" and second_type == "author":
                self.paper_authors[edge[0]].append(edge[1])
                self.author_papers[edge[1]].append(edge[0])


    def generate_walks(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        # paper - author
        for paper in self.paper_authors:
            p0 = paper
            for j in xrange(0, numwalks):  # num walks
                outline = p0
                for i in xrange(0, walklength):
                    authors = self.paper_authors[paper]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                    papers = self.author_papers[author]
                    paper = random.sample(papers, 1).pop()
                    outline += " " + paper
                outfile.write(outline + "\n")
                paper = p0

        # author - paper
        for author in self.author_papers:
            a0 = author
            for j in xrange(0, numwalks):  # num walks
                outline = a0
                for i in xrange(0, walklength):
                    papers = self.author_papers[author]
                    paper = random.sample(papers, 1).pop()
                    outline += " " + paper
                    authors = self.paper_authors[paper]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                outfile.write(outline + "\n")
                author = a0

        # field - author
        for field in self.field_authors:
            f0 = field
            for j in xrange(0, numwalks):  # num walks
                outline = f0
                for i in xrange(0, walklength):
                    authors = self.field_authors[field]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                    fields = self.author_fields[author]
                    field = random.sample(fields, 1).pop()
                    outline += " " + field
                outfile.write(outline + "\n")
                field = f0

        # author - field
        for author in self.author_fields:
            a0 = author
            for j in xrange(0, numwalks):  # num walks
                outline = a0
                for i in xrange(0, walklength):
                    fields = self.author_fields[author]
                    field = random.sample(fields, 1).pop()
                    outline += " " + field
                    authors = self.field_authors[field]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                outfile.write(outline + "\n")
                author = a0

    def generate_walks_2(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        # paper - author - field - author - paper
        for paper in self.paper_authors:
            p0 = paper
            for j in xrange(0, numwalks):  # num walks
                outline = p0
                for i in xrange(0, walklength):
                    authors = self.paper_authors[paper]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                    fields = self.author_fields[author]
                    if fields:
                        field = random.sample(fields, 1).pop()
                        outline += " " + field
                        authors = self.field_authors[field]
                        author = random.sample(authors,1).pop()
                        outline += " " + author
                        papers = self.author_papers[author]
                        if papers:
                            paper = random.sample(papers, 1).pop()
                            outline += " " + paper
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                paper = p0

        # author - field - author - paper - author
        for author in self.author_fields:
            a0 = author
            for j in xrange(0, numwalks):  # num walks
                outline = a0
                for i in xrange(0, walklength):
                    fields = self.author_fields[author]
                    field = random.sample(fields, 1).pop()
                    outline += " " + field
                    authors = self.field_authors[field]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                    papers = self.author_papers[author]
                    if papers:
                        paper = random.sample(papers, 1).pop()
                        outline += " " + paper
                        authors = self.paper_authors[paper]
                        if authors:
                            author = random.sample(authors, 1).pop()
                            outline += " " + author
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                author = a0

        # field - author - paper - author - field
        for field in self.field_authors:
            f0 = field
            for j in xrange(0, numwalks):  # num walks
                outline = f0
                for i in xrange(0, walklength):
                    authors = self.field_authors[field]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                    papers = self.author_papers[author]
                    if papers:
                        paper = random.sample(papers, 1).pop()
                        outline += " " + paper
                        authors = self.paper_authors[paper]
                        author = random.sample(authors, 1).pop()
                        outline += " " + author
                        fields = self.author_fields[author]
                        if fields:
                            field = random.sample(fields, 1).pop()
                            outline += " " + field
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                field = f0


        # author - paper - author - field - author
        for author in self.author_papers:
            a0 = author
            for j in xrange(0, numwalks):  # num walks
                outline = a0
                for i in xrange(0, walklength):
                    papers = self.author_papers[author]
                    paper = random.sample(papers, 1).pop()
                    outline += " " + paper
                    authors = self.paper_authors[paper]
                    author = random.sample(authors, 1).pop()
                    outline += " " + author
                    fields = self.author_fields[author]
                    if fields:
                        field = random.sample(fields, 1).pop()
                        outline += " " + field
                        authors = self.field_authors[field]
                        if authors:
                            author = random.sample(authors, 1).pop()
                            outline += " " + author
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                author = a0


class MetaPathGeneratorBio:
    def __init__(self, seed):
        self.gene_genes = defaultdict(list)
        self.gene_diseases = defaultdict(list)
        self.disease_genes = defaultdict(list)
        self.drug_genes = defaultdict(list)
        self.gene_drugs = defaultdict(list)
        random.seed(seed)

    def read_data(self, G):
        for edge in G.edges():
            first_type = edge[0].split("_")[0]
            second_type = edge[1].split("_")[0]
            if first_type == "disease" and second_type == "gene":
                self.disease_genes[edge[0]].append(edge[1])
                self.gene_diseases[edge[1]].append(edge[0])
            elif first_type == "drug" and second_type == "gene":
                self.drug_genes[edge[0]].append(edge[1])
                self.gene_drugs[edge[1]].append(edge[0])
            elif first_type == "gene" and second_type == "gene":
                self.gene_genes[edge[0]].append(edge[1])
                self.gene_genes[edge[1]].append(edge[0])
            elif first_type == "gene" and second_type == "drug":
                self.drug_genes[edge[1]].append(edge[0])
                self.gene_drugs[edge[0]].append(edge[1])
            elif first_type == "gene" and second_type == "disease":
                self.disease_genes[edge[1]].append(edge[0])
                self.gene_diseases[edge[0]].append(edge[1])


    def generate_walks(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        all_genes = set(self.gene_genes.keys())

        # gene-gene paths
        for gene in self.gene_genes:
            g0 = gene
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    genes = self.gene_genes[gene]
                    gene = random.sample(genes, 1).pop()
                    outline += " " + gene
                outfile.write(outline + "\n")
                gene = g0

        # disease-gene-disease paths
        for disease in self.disease_genes:
            di0 = disease
            for j in xrange(0, numwalks):  # num walks
                outline = di0
                for i in xrange(0, walklength):
                    genes = self.disease_genes[disease]
                    gene = random.sample(genes, 1).pop()
                    all_genes.add(gene)
                    outline += " " + gene
                    diseases = self.gene_diseases[gene]
                    disease = random.sample(diseases, 1).pop()
                    outline += " " + disease
                outfile.write(outline + "\n")
                disease = di0

        # drug-gene-drug paths
        for drug in self.drug_genes:
            dr0 = drug
            for j in xrange(0, numwalks):  # num walks
                outline = dr0
                for i in xrange(0, walklength):
                    genes = self.drug_genes[drug]
                    gene = random.sample(genes, 1).pop()
                    all_genes.add(gene)
                    outline += " " + gene
                    drugs = self.gene_drugs[gene]
                    drug = random.sample(drugs, 1).pop()
                    outline += " " + drug
                outfile.write(outline + "\n")
                drug = dr0


        for gene in self.gene_diseases:
            g0 = gene
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    diseases = self.gene_diseases[gene]
                    disease = random.sample(diseases, 1).pop()
                    outline += " " + disease
                    genes = self.disease_genes[disease]
                    gene = random.sample(genes, 1).pop()
                    all_genes.add(gene)
                    outline += " " + gene

                outfile.write(outline + "\n")
                gene = g0

        for gene in self.gene_drugs:
            g0 = gene
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    drugs = self.gene_drugs[gene]
                    drug = random.sample(drugs, 1).pop()
                    outline += " " + drug
                    genes = self.drug_genes[drug]
                    gene = random.sample(genes, 1).pop()
                    outline += " " + gene

                outfile.write(outline + "\n")
                gene = g0

        all_genes_2 = set(self.gene_diseases.keys() + self.gene_drugs.keys() + self.gene_genes.keys())
        d = all_genes_2.difference(all_genes)

        print("Elements not used:", d)
        for g in d:
            outfile.write(g + "\n")

    def generate_walks_2(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        all_genes = set(self.gene_genes.keys() + self.gene_diseases.keys() + self.gene_drugs.keys())
        # gene - disease - gene - drug - gene
        for gene in all_genes:
            g0 = gene
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    diseases = self.gene_diseases[gene]
                    if diseases:
                        disease = random.sample(diseases, 1).pop()
                        outline += " " + disease
                        genes = self.disease_genes[disease]
                        gene = random.sample(genes, 1).pop()
                        outline += " " + gene
                        drugs = self.gene_drugs[gene]
                        if drugs:
                            drug = random.sample(drugs, 1).pop()
                            outline += " " + drug
                            genes = self.drug_genes[drug]
                            gene = random.sample(genes, 1).pop()
                            outline += " " + gene
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                gene = g0

        # drug - gene - disease - gene - drug
        for drug in self.drug_genes:
            dr0 = drug
            for j in xrange(0, numwalks):  # num walks
                outline = dr0
                for i in xrange(0, walklength):
                    genes = self.drug_genes[drug]
                    gene = random.sample(genes, 1).pop()
                    outline += " " + gene
                    diseases = self.gene_diseases[gene]
                    if diseases:
                        disease = random.sample(diseases, 1).pop()
                        outline += " " + disease
                        genes = self.disease_genes[disease]
                        gene = random.sample(genes, 1).pop()
                        outline += " " + gene
                        drugs = self.gene_drugs[gene]
                        if drugs:
                            drug = random.sample(drugs, 1).pop()
                            outline += " " + drug
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                drug = dr0

        # disease - gene - drug - gene - disease
        for disease in self.disease_genes:
            di0 = disease
            for j in xrange(0, numwalks):  # num walks
                outline = di0
                for i in xrange(0, walklength):
                    genes = self.disease_genes[disease]
                    gene = random.sample(genes, 1).pop()
                    outline += " " + gene
                    drugs = self.gene_drugs[gene]
                    if drugs:

                        drug = random.sample(drugs, 1).pop()
                        outline += " " + drug
                        genes = self.drug_genes[drug]
                        gene = random.sample(genes, 1).pop()
                        outline += " " + gene
                        diseases = self.gene_diseases[gene]
                        if diseases:
                            disease = random.sample(diseases, 1).pop()
                            outline += " " + disease
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                disease = di0






#python py4genMetaPaths.py 1000 100 net_aminer output.aminer.w1000.l100.txt
#python py4genMetaPaths.py 1000 100 net_dbis   output.dbis.w1000.l100.txt

# dirpath = "net_aminer"
# OR 
# dirpath = "net_dbis"

# numwalks = int(sys.argv[1])
# walklength = int(sys.argv[2])
#
# dirpath = sys.argv[3]
# outfilename = sys.argv[4]
#
# def main():
#     mpg = MetaPathGeneratorWomen()
#     mpg.read_data(dirpath)
#     mpg.generate_random_wew(outfilename, numwalks, walklength)
#
#
# if __name__ == "__main__":
#     main()
#
