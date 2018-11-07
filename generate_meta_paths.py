import sys
import os
import random
from collections import Counter
from collections import defaultdict

class MetaPathGenerator:
    def __init__(self):
        self.id_author = dict()
        self.id_conf = dict()
        self.author_coauthorlist = dict()
        self.conf_authorlist = dict()
        self.author_conflist = dict()
        self.paper_author = dict()
        self.author_paper = dict()
        self.conf_paper = dict()
        self.paper_conf = dict()

    def read_data(self, dirpath):
        with open(dirpath + "/index_author.tsv") as adictfile:
            for line in adictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    self.id_author[toks[0]] = toks[1].replace(" ", "")

        #print "#authors", len(self.id_author)

        with open(dirpath + "/index_venue.tsv") as cdictfile:
            for line in cdictfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    newconf = toks[1].replace(" ", "")
                    self.id_conf[toks[0]] = newconf

        #print "#conf", len(self.id_conf)

        with open(dirpath + "/author_paper.tsv") as pafile:
            for line in pafile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    a, p = toks[0], toks[1]
                    if p not in self.paper_author:
                        self.paper_author[p] = []
                    self.paper_author[p].append(a)
                    if a not in self.author_paper:
                        self.author_paper[a] = []
                    self.author_paper[a].append(p)

        with open(dirpath + "/paper_venue.tsv") as pcfile:
            for line in pcfile:
                toks = line.strip().split("\t")
                if len(toks) == 2:
                    p, c = toks[0], toks[1]
                    self.paper_conf[p] = c 
                    if c not in self.conf_paper:
                        self.conf_paper[c] = []
                    self.conf_paper[c].append(p)

        sumpapersconf, sumauthorsconf = 0, 0
        conf_authors = dict()
        for conf in self.conf_paper:
            papers = self.conf_paper[conf]
            sumpapersconf += len(papers)
            for paper in papers:
                if paper in self.paper_author:
                    authors = self.paper_author[paper]
                    sumauthorsconf += len(authors)

        print "#confs  ", len(self.conf_paper)
        print "#papers ", sumpapersconf,  "#papers per conf ", sumpapersconf / len(self.conf_paper)
        print "#authors", sumauthorsconf, "#authors per conf", sumauthorsconf / len(self.conf_paper)


    def generate_random_aca(self, outfilename, numwalks, walklength):
        for conf in self.conf_paper:
            self.conf_authorlist[conf] = []
            for paper in self.conf_paper[conf]:
                if paper not in self.paper_author: continue
                for author in self.paper_author[paper]:
                    self.conf_authorlist[conf].append(author)
                    if author not in self.author_conflist:
                        self.author_conflist[author] = []
                    self.author_conflist[author].append(conf)
        #print "author-conf list done"

        outfile = open(dirpath + "/" + outfilename, 'w')
        for conf in self.conf_authorlist:
            conf0 = conf
            for j in xrange(0, numwalks ): #wnum walks
                outline = self.id_conf[conf0]
                for i in xrange(0, walklength):
                    authors = self.conf_authorlist[conf]
                    numa = len(authors)
                    authorid = random.randrange(numa)
                    author = authors[authorid]
                    outline += " " + self.id_author[author]
                    confs = self.author_conflist[author]
                    numc = len(confs)
                    confid = random.randrange(numc)
                    conf = confs[confid]
                    outline += " " + self.id_conf[conf]
                outfile.write(outline + "\n")
        outfile.close()

class MetaPathGeneratorWomen:
    def __init__(self):
        self.woman_events = defaultdict(list)
        self.event_women = defaultdict(list)

    def read_data(self, dirpath):
        with open(dirpath + "/women.txt") as adictfile:
            header = adictfile.readline()
            for line in adictfile:
                toks = line.strip().split(" ") # event woman
                if len(toks) == 2:
                    self.event_women["e"+toks[0]].append("w"+toks[1])
                    self.woman_events["w"+toks[1]].append("e"+toks[0])

    def generate_random_wew(self, outfilename, numwalks, walklength):
        # woman-event-woman
        outfile = open(dirpath + "/" + outfilename, 'w')
        for woman in self.woman_events:
            w0 = woman
            for j in xrange(0, numwalks ): #wnum walks
                outline = w0
                for i in xrange(0, walklength):
                    events = self.woman_events[woman]
                    num_events = len(events)
                    event_id= random.randrange(num_events)
                    event = events[event_id]
                    outline += " " + event
                    women = self.event_women[event]
                    num_women = len(women)
                    woman_id = random.randrange(num_women)
                    woman = women[woman_id]
                    outline += " " + woman
                outfile.write(outline + "\n")
        outfile.close()


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

        print(len(all_genes))
        all_genes_2 = set(self.gene_diseases.keys() + self.gene_drugs.keys() + self.gene_genes.keys())
        print(len(all_genes_2))
        d = all_genes_2.difference(all_genes)

        print("Elements not used:", d)
        for g in d:
            outfile.write(g + "\n")





    def generate_random_di_g_di(self, outfilename, numwalks, walklength):
        # disease-gene-disease
        outfile = open(outfilename, 'w')
        for disease in self.disease_genes:
            di0 = disease
            for j in xrange(0, numwalks):  # num walks
                outline = di0
                for i in xrange(0, walklength):
                    genes = self.disease_genes[disease]
                    gene = random.sample(genes, 1).pop()
                    outline += " " + gene
                    diseases = self.gene_diseases[gene]
                    disease = random.sample(diseases, 1).pop()
                    outline += " " + disease
                outfile.write(outline + "\n")

    def generate_random_g_di_g(self, outfilename, numwalks, walklength):
        # gene-disease-gene
        # TODO preverit ce se vse bolezni pojavijo
        outfile = open(outfilename, 'w')
        all_genes = set(self.gene_diseases.keys() + self.gene_drugs.keys() + self.gene_genes.keys())
        for gene in all_genes:
            g0 = gene
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    if gene in self.gene_diseases:
                        diseases = self.gene_diseases[gene]
                        disease = random.sample(diseases, 1).pop()
                        outline += " " + disease
                        genes = self.disease_genes[disease]
                        gene = random.sample(genes, 1).pop()
                        outline += " " + gene
                    else:
                        outline += " " + gene + " " + gene

                outfile.write(outline + "\n")

    def generate_random_g_dr_g(self, outfilename, numwalks, walklength):
        # gene-drug-gene
        outfile = open(outfilename, 'w')
        all_genes = set(self.gene_diseases.keys() + self.gene_drugs.keys() + self.gene_genes.keys())
        for gene in all_genes:
            g0 = gene
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    if gene in self.gene_drugs:
                        drugs = self.gene_drugs[gene]
                        drug = random.sample(drugs, 1).pop()
                        outline += " " + drug
                        genes = self.drug_genes[drug]
                        gene = random.sample(genes, 1).pop()
                        outline += " " + gene
                    else:
                        outline += " " + gene + " " + gene

                outfile.write(outline + "\n")

    def generate_random_dr_g_dr(self, outfilename, numwalks, walklength):
        # drug-gene-drug
        outfile = open(outfilename, 'w')
        for drug in self.drug_genes:
            di0 = drug
            for j in xrange(0, numwalks):  # num walks
                outline = di0
                for i in xrange(0, walklength):
                    genes = self.drug_genes[drug]
                    gene = random.sample(genes, 1).pop()
                    outline += " " + gene
                    drugs = self.gene_drugs[gene]
                    drug = random.sample(drugs, 1).pop()
                    outline += " " + drug
                outfile.write(outline + "\n")

    #
    # def generate_random_wew(self, outfilename, numwalks, walklength):
    #     # woman-event-woman
    #     outfile = open(dirpath + "/" + outfilename, 'w')
    #     for woman in self.woman_events:
    #         w0 = woman
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = w0
    #             for i in xrange(0, walklength):
    #                 events = self.woman_events[woman]
    #                 num_events = len(events)
    #                 event_id = random.randrange(num_events)
    #                 event = events[event_id]
    #                 outline += " " + event
    #                 women = self.event_women[event]
    #                 num_women = len(women)
    #                 woman_id = random.randrange(num_women)
    #                 woman = women[woman_id]
    #                 outline += " " + woman
    #             outfile.write(outline + "\n")
    #     outfile.close()


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
