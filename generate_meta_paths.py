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
                    if fields:
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
                            authors = self.field_authors[field]
                            author = random.sample(authors, 1).pop()
                            outline += " " + author
                        else:
                            break
                    else:
                        break
                outfile.write(outline + "\n")
                author = a0

class MetaPathGeneratorImdb:
    def __init__(self, seed):
        self.title_actors = defaultdict(list)
        self.actor_titles = defaultdict(list)
        self.title_genres = defaultdict(list)
        self.genre_titles = defaultdict(list)
        self.title_crews = defaultdict(list)
        self.crew_titles = defaultdict(list)
        random.seed(seed)

    def read_data(self, G):
        for edge in G.edges():
            first_type = edge[0].split("_")[0]
            second_type = edge[1].split("_")[0]
            if first_type == "title" and second_type == "actor":
                self.title_actors[edge[0]].append(edge[1])
                self.actor_titles[edge[1]].append(edge[0])
            elif first_type == "actor" and second_type == "title":
                self.actor_titles[edge[0]].append(edge[1])
                self.title_actors[edge[1]].append(edge[0])
            elif first_type == "title" and second_type == "genre":
                self.title_genres[edge[0]].append(edge[1])
                self.genre_titles[edge[1]].append(edge[0])
            elif first_type == "genre" and second_type == "title":
                self.genre_titles[edge[0]].append(edge[1])
                self.title_genres[edge[1]].append(edge[0])
            elif first_type == "title" and second_type == "crew":
                self.title_crews[edge[0]].append(edge[1])
                self.crew_titles[edge[1]].append(edge[0])
            elif first_type == "crew" and second_type == "title":
                self.crew_titles[edge[0]].append(edge[1])
                self.title_crews[edge[1]].append(edge[0])

    def generate_walks(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        # actor - title
        for actor in self.actor_titles:
            a0 = actor
            for j in xrange(0, numwalks):  # num walks
                outline = a0
                for i in xrange(0, walklength):
                    titles = self.actor_titles[actor]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    actors = self.title_actors[title]
                    actor = random.sample(actors, 1).pop()
                    outline += " " + actor
                outfile.write(outline + "\n")
                actor = a0

        # title - actor
        for title in self.title_actors:
            t0 = title
            for j in xrange(0, numwalks):  # num walks
                outline = t0
                for i in xrange(0, walklength):
                    actors = self.title_actors[title]
                    actor = random.sample(actors, 1).pop()
                    outline += " " + actor
                    titles = self.actor_titles[actor]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                outfile.write(outline + "\n")
                title = t0

        # genre - title
        for genre in self.genre_titles:
            g0 = genre
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    titles = self.genre_titles[genre]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    genres = self.title_genres[title]
                    genre = random.sample(genres, 1).pop()
                    outline += " " + genre
                outfile.write(outline + "\n")
                genre = g0

        # title - genre
        for title in self.title_genres:
            t0 = title
            for j in xrange(0, numwalks):  # num walks
                outline = t0
                for i in xrange(0, walklength):
                    genres = self.title_genres[title]
                    genre = random.sample(genres, 1).pop()
                    outline += " " + genre
                    titles = self.genre_titles[genre]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                outfile.write(outline + "\n")
                title = t0

        # crew - title
        for crew in self.crew_titles:
            c0 = crew
            for j in xrange(0, numwalks):  # num walks
                outline = c0
                for i in xrange(0, walklength):
                    titles = self.crew_titles[crew]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    crews = self.title_crews[title]
                    crew = random.sample(crews, 1).pop()
                    outline += " " + crew
                outfile.write(outline + "\n")
                crew = c0

        # title - crew
        for title in self.title_crews:
            t0 = title
            for j in xrange(0, numwalks):  # num walks
                outline = t0
                for i in xrange(0, walklength):
                    crews = self.title_crews[title]
                    crew = random.sample(crews, 1).pop()
                    outline += " " + crew
                    titles = self.crew_titles[crew]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                outfile.write(outline + "\n")
                title = t0

    def generate_walks_2(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        # actor - title - crew - title - actor
        for actor in self.actor_titles:
            a0 = actor
            for j in xrange(0, numwalks):  # num walks
                outline = a0
                for i in xrange(0, walklength):
                    titles = self.actor_titles[actor]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    crews = self.title_crews[title]
                    if crews:
                        crew = random.sample(crews, 1).pop()
                        outline += " " + crew
                        titles = self.crew_titles[crew]
                        title = random.sample(titles, 1).pop()
                        outline += " " + title
                        actors = self.title_actors[title]
                        if actors:
                            actor = random.sample(actors, 1).pop()
                            outline += " " + actor
                        else:
                            break
                else:
                    break
                outfile.write(outline + "\n")
                actor = a0


        # genre - title - crew - title - genre
        for genre in self.genre_titles:
            g0 = genre
            for j in xrange(0, numwalks):  # num walks
                outline = g0
                for i in xrange(0, walklength):
                    titles = self.genre_titles[genre]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    crews = self.title_crews[title]
                    if crews:
                        crew = random.sample(crews, 1).pop()
                        outline += " " + crew
                        titles = self.crew_titles[crew]
                        title = random.sample(titles, 1).pop()
                        outline += " " + title
                        genres = self.title_genres[title]
                        if genres:
                            genre = random.sample(genres, 1).pop()
                            outline += " " + genre
                        else:
                            break
                    else:
                        break
                outfile.write(outline + "\n")
                genre = g0

        # crew - title - actor - title - crew
        for crew in self.crew_titles:
            c0 = crew
            for j in xrange(0, numwalks):  # num walks
                outline = c0
                for i in xrange(0, walklength):
                    titles = self.crew_titles[crew]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    actors = self.title_actors[title]
                    if actors:
                        actor = random.sample(actors, 1).pop()
                        outline += " " + actor
                        titles = self.actor_titles[actor]
                        title = random.sample(titles, 1).pop()
                        outline += " " + title
                        crews = self.title_crews[title]
                        if crews:
                            crew = random.sample(crews, 1).pop()
                            outline += " " + crew
                        else:
                            break
                    else:
                        break
                outfile.write(outline + "\n")
                crew = c0

        # title - actor - title - genre - title
        for title in self.title_actors:
            t0 = title
            for j in xrange(0, numwalks):  # num walks
                outline = t0
                for i in xrange(0, walklength):
                    actors = self.title_actors[title]
                    actor = random.sample(actors, 1).pop()
                    outline += " " + actor
                    titles = self.actor_titles[actor]
                    title = random.sample(titles, 1).pop()
                    outline += " " + title
                    genres = self.title_genres[title]
                    if genres:
                        genre = random.sample(genres, 1).pop()
                        outline += " " + genre
                        titles = self.genre_titles[genre]
                        title = random.sample(titles, 1).pop()
                        outline += " " + title
                    else:
                        break
                    outfile.write(outline + "\n")
                title = t0


class MetaPathGeneratorAmazon:
    def __init__(self, seed):
        self.product_products = defaultdict(list)
        self.user_products = defaultdict(list)
        self.product_users = defaultdict(list)
        self.product_categories = defaultdict(list)
        self.category_products = defaultdict(list)
        random.seed(seed)

    def read_data(self, G):
        for edge in G.edges():
            first_type = edge[0].split("_")[0]
            second_type = edge[1].split("_")[0]
            if first_type == "product" and second_type == "category":
                self.product_categories[edge[0]].append(edge[1])
                self.category_products[edge[1]].append(edge[0])
            elif first_type == "category" and second_type == "product":
                self.category_products[edge[0]].append(edge[1])
                self.product_categories[edge[1]].append(edge[0])
            elif first_type == "product" and second_type == "product":
                self.product_products[edge[0]].append(edge[1])
                self.product_products[edge[1]].append(edge[0])
            elif first_type == "product" and second_type == "user":
                self.product_users[edge[0]].append(edge[1])
                self.user_products[edge[1]].append(edge[0])
            elif first_type == "user" and second_type == "product":
                self.user_products[edge[0]].append(edge[1])
                self.product_users[edge[1]].append(edge[0])


    def generate_walks(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        # product-product paths
        for product in self.product_products:
            p0 = product
            for j in xrange(0, numwalks):  # num walks
                outline = p0
                for i in xrange(0, walklength):
                    products = self.product_products[product]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                outfile.write(outline + "\n")
                product = p0

        # user-product-user paths
        for user in self.user_products:
            u0 = user
            for j in xrange(0, numwalks):  # num walks
                outline = u0
                for i in xrange(0, walklength):
                    products = self.user_products[user]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                    users = self.product_users[product]
                    user = random.sample(users, 1).pop()
                    outline += " " + user
                outfile.write(outline + "\n")
                user = u0

        # product-user-product paths
        for product in self.product_users:
            p0 = product
            for j in xrange(0, numwalks):  # num walks
                outline = p0
                for i in xrange(0, walklength):
                    users = self.product_users[product]
                    user = random.sample(users, 1).pop()
                    outline += " " + user
                    products = self.user_products[user]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                outfile.write(outline + "\n")
                product = p0

        # category-product-category paths
        for category in self.category_products:
            c0 = category
            for j in xrange(0, numwalks):  # num walks
                outline = c0
                for i in xrange(0, walklength):
                    products = self.category_products[category]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                    categories = self.product_categories[product]
                    category = random.sample(categories, 1).pop()
                    outline += " " + category
                outfile.write(outline + "\n")
                category = c0

        # product-category-product paths
        for product in self.product_categories:
            p0 = product
            for j in xrange(0, numwalks):  # num walks
                outline = p0
                for i in xrange(0, walklength):
                    categories = self.product_categories[product]
                    category = random.sample(categories, 1).pop()
                    outline += " " + category
                    products = self.category_products[category]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                outfile.write(outline + "\n")
                product = p0


    def generate_walks_2(self, outfilename, numwalks, walklength):
        outfile = open(outfilename, 'w')

        all_products = set(self.product_categories.keys() + self.product_users.keys() + self.product_products.keys())
        # product - category - product - user - product
        for product in all_products:
            p0 = product
            for j in xrange(0, numwalks):  # num walks
                outline = p0
                for i in xrange(0, walklength):
                    categories = self.product_categories[product]
                    if categories:
                        category = random.sample(categories, 1).pop()
                        outline += " " + category
                        products = self.category_products[category]
                        product = random.sample(products, 1).pop()
                        outline += " " + product
                        users = self.product_users[product]
                        if users:
                            user = random.sample(users, 1).pop()
                            outline += " " + user
                            products = self.user_products[user]
                            product = random.sample(products, 1).pop()
                            outline += " " + product
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                product = p0

        # category - product - user - product - category
        for category in self.category_products:
            c0 = category
            for j in xrange(0, numwalks):  # num walks
                outline = c0
                for i in xrange(0, walklength):
                    products = self.category_products[category]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                    users = self.product_users[product]
                    if users:
                        user = random.sample(users, 1).pop()
                        outline += " " + user
                        products = self.user_products[user]
                        product = random.sample(products, 1).pop()
                        outline += " " + product
                        categories = self.product_categories[product]
                        if categories:
                            category = random.sample(categories, 1).pop()
                            outline += " " + category
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                category = c0

        # user - product - category - product - user
        for user in self.user_products:
            u0 = user
            for j in xrange(0, numwalks):  # num walks
                outline = u0
                for i in xrange(0, walklength):
                    products = self.user_products[user]
                    product = random.sample(products, 1).pop()
                    outline += " " + product
                    categories = self.product_categories[product]
                    if categories:
                        category = random.sample(categories, 1).pop()
                        outline += " " + category
                        products = self.category_products[category]
                        product = random.sample(products, 1).pop()
                        outline += " " + product
                        users = self.product_users[product]
                        if users:
                            user = random.sample(users, 1).pop()
                            outline += " " + user
                        else:
                            break
                    else:
                        break

                outfile.write(outline + "\n")
                user = u0


class MetaPathGeneratorYelp:
    def __init__(self, seed):
        self.user_users = defaultdict(list)
        self.user_businesses = defaultdict(list)
        self.business_users = defaultdict(list)
        self.business_categories = defaultdict(list)
        self.category_businesses = defaultdict(list)
        random.seed(seed)

    def read_data(self, G):
        for edge in G.edges():
            first_type = edge[0].split("_")[0]
            second_type = edge[1].split("_")[0]
            if first_type == "business" and second_type == "category":
                self.business_categories[edge[0]].append(edge[1])
                self.category_businesses[edge[1]].append(edge[0])
            elif first_type == "category" and second_type == "business":
                self.category_businesses[edge[0]].append(edge[1])
                self.business_categories[edge[1]].append(edge[0])
            elif first_type == "user" and second_type == "user":
                self.user_users[edge[0]].append(edge[1])
                self.user_users[edge[1]].append(edge[0])
            elif first_type == "business" and second_type == "user":
                self.business_users[edge[0]].append(edge[1])
                self.user_businesses[edge[1]].append(edge[0])
            elif first_type == "user" and second_type == "business":
                self.user_businesses[edge[0]].append(edge[1])
                self.business_users[edge[1]].append(edge[0])

    #
    # def generate_walks(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #
    #     # user-user paths
    #     for user in self.user_users:
    #         u0 = user
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = u0
    #             for i in xrange(0, walklength):
    #                 users = self.user_users[user]
    #                 user = random.sample(users, 1).pop()
    #                 outline += " " + user
    #             outfile.write(outline + "\n")
    #             user = u0
    #
    #     # user-business-user paths
    #     for user in self.user_products:
    #         u0 = user
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = u0
    #             for i in xrange(0, walklength):
    #                 products = self.user_products[user]
    #                 product = random.sample(products, 1).pop()
    #                 outline += " " + product
    #                 users = self.product_users[product]
    #                 user = random.sample(users, 1).pop()
    #                 outline += " " + user
    #             outfile.write(outline + "\n")
    #             user = u0
    #
    #     # business-user-business paths
    #     for product in self.product_users:
    #         p0 = product
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = p0
    #             for i in xrange(0, walklength):
    #                 users = self.product_users[product]
    #                 user = random.sample(users, 1).pop()
    #                 outline += " " + user
    #                 products = self.user_products[user]
    #                 product = random.sample(products, 1).pop()
    #                 outline += " " + product
    #             outfile.write(outline + "\n")
    #             product = p0
    #
    #     # category-product-category paths
    #     for category in self.category_products:
    #         c0 = category
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = c0
    #             for i in xrange(0, walklength):
    #                 products = self.category_products[category]
    #                 product = random.sample(products, 1).pop()
    #                 outline += " " + product
    #                 categories = self.product_categories[product]
    #                 category = random.sample(categories, 1).pop()
    #                 outline += " " + category
    #             outfile.write(outline + "\n")
    #             category = c0
    #
    #     # product-category-product paths
    #     for product in self.product_categories:
    #         p0 = product
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = p0
    #             for i in xrange(0, walklength):
    #                 categories = self.product_categories[product]
    #                 category = random.sample(categories, 1).pop()
    #                 outline += " " + category
    #                 products = self.category_products[category]
    #                 product = random.sample(products, 1).pop()
    #                 outline += " " + product
    #             outfile.write(outline + "\n")
    #             product = p0
    #
    #
    # def generate_walks_2(self, outfilename, numwalks, walklength):
    #     outfile = open(outfilename, 'w')
    #
    #     all_products = set(self.product_categories.keys() + self.product_users.keys() + self.product_products.keys())
    #     # product - category - product - user - product
    #     for product in all_products:
    #         p0 = product
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = p0
    #             for i in xrange(0, walklength):
    #                 categories = self.product_categories[product]
    #                 if categories:
    #                     category = random.sample(categories, 1).pop()
    #                     outline += " " + category
    #                     products = self.category_products[category]
    #                     product = random.sample(products, 1).pop()
    #                     outline += " " + product
    #                     users = self.product_users[product]
    #                     if users:
    #                         user = random.sample(users, 1).pop()
    #                         outline += " " + user
    #                         products = self.user_products[user]
    #                         product = random.sample(products, 1).pop()
    #                         outline += " " + product
    #                     else:
    #                         break
    #                 else:
    #                     break
    #
    #             outfile.write(outline + "\n")
    #             product = p0
    #
    #     # category - product - user - product - category
    #     for category in self.category_products:
    #         c0 = category
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = c0
    #             for i in xrange(0, walklength):
    #                 products = self.category_products[category]
    #                 product = random.sample(products, 1).pop()
    #                 outline += " " + product
    #                 users = self.product_users[product]
    #                 if users:
    #                     user = random.sample(users, 1).pop()
    #                     outline += " " + user
    #                     products = self.user_products[user]
    #                     product = random.sample(products, 1).pop()
    #                     outline += " " + product
    #                     categories = self.product_categories[product]
    #                     if categories:
    #                         category = random.sample(categories, 1).pop()
    #                         outline += " " + category
    #                     else:
    #                         break
    #                 else:
    #                     break
    #
    #             outfile.write(outline + "\n")
    #             category = c0
    #
    #     # user - product - category - product - user
    #     for user in self.user_products:
    #         u0 = user
    #         for j in xrange(0, numwalks):  # num walks
    #             outline = u0
    #             for i in xrange(0, walklength):
    #                 products = self.user_products[user]
    #                 product = random.sample(products, 1).pop()
    #                 outline += " " + product
    #                 categories = self.product_categories[product]
    #                 if categories:
    #                     category = random.sample(categories, 1).pop()
    #                     outline += " " + category
    #                     products = self.category_products[category]
    #                     product = random.sample(products, 1).pop()
    #                     outline += " " + product
    #                     users = self.product_users[product]
    #                     if users:
    #                         user = random.sample(users, 1).pop()
    #                         outline += " " + user
    #                     else:
    #                         break
    #                 else:
    #                     break
    #
    #             outfile.write(outline + "\n")
    #             user = u0


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
