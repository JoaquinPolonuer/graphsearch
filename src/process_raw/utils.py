import json
import numpy as np
import pandas as pd

MAX_ENTRIES = 24


def add_summary_amazon(row) -> str:
    if row["type"] == "brand":
        return f"brand name: {row['brand_name']}"
    if row["type"] == "category":
        return f"category name: {row['category_name']}"
    if row["type"] == "color":
        return f"color name: {row['color_name']}"

    summary = f"- product: {row['title']}\n"
    if row["brand"]:
        summary += f"- brand: {row['brand']}\n"
    if row["details"]:
        try:
            dimensions, weight = row["details"].product_dimensions.split(" ; ")
            summary += f"- dimensions: {dimensions}\n- weight: {weight}\n"
        except:
            pass
    if row["description"]:
        row_description = eval(row["description"])
        description = " ".join(row_description).strip(" ")
        if description:
            summary += f"- description: {description}\n"

    feature_text = "- features: \n"
    if row["feature"]:
        row_feature = eval(row["feature"])
        for feature_idx, feature in enumerate(row_feature):
            if feature and "asin" not in feature.lower():
                feature_text += f"#{feature_idx + 1}: {feature}\n"
    else:
        feature_text = ""

    # We have a problem here
    # if row["review"]:
    #     review_text = "- reviews: \n"
    #     scores = [
    #         0 if pd.isnull(review["vote"]) else int(review["vote"].replace(",", ""))
    #         for review in row["review"]
    #     ]
    #     ranks = np.argsort(-np.array(scores))
    #     for i, review_idx in enumerate(ranks):
    #         review = row["review"][review_idx]
    #         review_text += f'#{review_idx + 1}:\nsummary: {review["summary"]}\ntext: "{review["reviewText"]}"\n'
    #         if i > MAX_ENTRIES:
    #             break
    # else:
    #     review_text = ""

    # if row["qa"]:
    #     row_qa = eval(row["qa"])
    #     qa_text = "- Q&A: \n"
    #     for qa_idx, qa in enumerate(row_qa):
    #         qa_text += f'#{qa_idx + 1}:\nquestion: "{qa["question"]}"\nanswer: "{qa["answer"]}"\n'
    #         if qa_idx > MAX_ENTRIES:
    #             break
    # else:
    #     qa_text = ""

    summary += feature_text  # + review_text + qa_text

    # if add_rel:
    #     summary += self.get_rel_info(idx)
    # if compact:
    #     summary = compact_text(summary)
    return summary
