import os


def build_model_folder(path, name):
    os.makedirs(os.path.abspath(os.sep.join([path, name])), exist_ok=True)


def store_recommendation(recommendations, path):
    with open(path, "w", encoding="utf-8") as file:
        for user, items in recommendations.items():
            for item, score in items:
                file.write(f"{user}\t{item}\t{score}\n")
