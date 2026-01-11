import os 
import  faiss
import lightgbm as lgb
import pickle 
import pandas as pd 
import numpy as np 
from flask import Flask , request , jsonify , render_template


## paths  setups

Base_dir = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(Base_dir , "artifacts")

Faiss_path = os.path.join(ARTIFACTS_DIR,"faiss.index")
Ranker_path = os.path.join(ARTIFACTS_DIR , "lgbm_ranker.txt")
Data_path = os.path.join(ARTIFACTS_DIR , "recommendation.pkl")



index = faiss.read_index(Faiss_path)
ranker = lgb.Booster(model_file = Ranker_path)


with open(Data_path,'rb') as f :
    data_bundle = pickle.load(f)
    

train_df = data_bundle["train_df"]
train_embeddings = data_bundle["train_embedding"]
FEATURES = data_bundle["features"]

print("Artifacts load successfully")
print(f"Products in mode: {len(train_df)}")


app = Flask(__name__,template_folder="templates",static_folder="static")



def recommend(product_name , top_n= 5):
    if product_name  not in train_df["product_name"].values:
        return []
    
    query_index = train_df.index[train_df["product_name"]== product_name][0]
    
    scores , indices = index.search(train_embeddings[query_index].reshape(1,-1),
                                    top_n+1)
    
    recommendations = []
    
    for idx in indices[0]:
        idx = int(idx)
        if idx == query_index :
            continue
        
        product = train_df.iloc[idx]
        
        recommendations.append({
            "product_image_url_jpeg" :product["product_image_url_jpeg"],
            "product_name" : product["product_name"],
            "product_brand": product["product_brand"],
            "product_category":product["product_category"],
            "product_price" : float(product["product_price"]),
            "product_reviews_count" : int(product["product_reviews_count"]),
            "rating_for_model" : float(product["rating_for_model"])
            
        })
            
    return recommendations[:top_n]


@app.route("/")
def home():
    product_list = train_df["product_name"].sample(20).tolist()
    return render_template("index.html", products=product_list)


@app.route("/search", methods=["POST"])
def search():
    product_name = request.form.get("product_name")

    recommendations = recommend(product_name, top_n=5)

    return render_template(
        "results.html",
        product_name=product_name,
        recommendations=recommendations
    )



@app.route("/recommend", methods=["POST"])
def recommend_api():
    payload = request.get_json()

    if not payload or "product_name" not in payload:
        return jsonify({
            "error": "product_name is required"
        }), 400

    product_name = payload["product_name"]
    top_n = payload.get("top_n", 5)

    recommendations = recommend(product_name, top_n)

    if not recommendations:
        return jsonify({
            "product": product_name,
            "recommendations": [],
            "message": "Product not found"
        }), 404

    return jsonify({
        "product": product_name,
        "recommendations": recommendations
    })
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
    


