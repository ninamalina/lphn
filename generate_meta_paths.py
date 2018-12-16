import sys
import os
import random
from collections import Counter
from collections import defaultdict

class MetaPathGeneratorSicris:
    def __init__(self, seed):
        self.field_authors = dict()
        self.author_fields = dict()
        self.paper_authors = dict()
        self.author_papers = dict()
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

    # def generate_walks(selfoutfilename, numwalks, walklength):





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

        # drug-gene-drug paths
        for drug in self.drug_genes:
            di0 = drug
            for j in xrange(0, numwalks):  # num walks
                outline = di0
                for i in xrange(0, walklength):
                    genes = self.drug_genes[drug]
                    gene = random.sample(genes, 1).pop()
                    all_genes.add(gene)
                    outline += " " + gene
                    drugs = self.gene_drugs[gene]
                    drug = random.sample(drugs, 1).pop()
                    outline += " " + drug
                outfile.write(outline + "\n")


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

        # drug - gene - disease - gene - drug
        for drug in self.drug_genes:
            di0 = drug
            for j in xrange(0, numwalks):  # num walks
                outline = di0
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
