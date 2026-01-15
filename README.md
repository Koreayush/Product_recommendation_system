# 🛍️ Product Recommendation System — Production Ready AI Service

---

## 🔹 Overview

This repository contains a **scalable, cloud‑deployed Product Recommendation System** built using **NLP, deep learning embeddings, and machine learning**. The service is exposed via a **Flask REST API**, containerized with **Docker**, and deployed on **AWS ECS** for reliable, scalable, and real‑world production usage.

The system leverages **semantic search using transformer embeddings** and **FAISS vector retrieval**, combined with a **LightGBM ranking model**, to deliver highly relevant product recommendations in real time.

---

## 🎯 Objective

The goal of this system is to:

* Provide personalized product recommendations
* Understand product text using NLP embeddings
* Perform fast similarity search using FAISS
* Rank recommendations using a machine learning model
* Serve predictions through a scalable Flask API
* Run reliably in production using Docker and AWS ECS

---

## 🧠 Model & Recommendation Approach

The recommendation pipeline consists of three main components:

### 1️⃣ Text Representation (NLP Layer)

* Uses **Sentence-Transformers (all-MiniLM)** to convert product descriptions into dense vector embeddings.
* Captures semantic meaning instead of simple keyword matching.

### 2️⃣ Fast Similarity Search (Retrieval Layer)

* Stores embeddings in **FAISS (Facebook AI Similarity Search)** for ultra‑fast nearest neighbor retrieval.
* Returns top-K most similar products based on user input or selected item.

### 3️⃣ Ranking Model (ML Layer)

* A **LightGBM model** re-ranks retrieved candidates based on learned relevance signals.
* Ensures business-friendly and high-quality recommendations.

---

## 🏗️ System Architecture

```plaintext
User / Frontend
       ↓ HTTP Request
Flask REST API (Gunicorn)
       ↓
Sentence-Transformers Embeddings
       ↓
FAISS Vector Search
       ↓
LightGBM Ranking Model
       ↓
Docker Container
       ↓
AWS ECS (Load Balanced)
```

---

## 🛠️ Technologies Used


# Software Tools required 

1. [Github Account](https://github.com)
2. [VS Code IDE](https://code.visualstudio.com/)
3. [AWS Account](https://aws.amazon.com)
4. [GitCLI](https://cli.github.com/)


# Create a new environment 

```bash
py -3.11 -m venv venv

```

**Backend & API**

* Flask
* Gunicorn (Production WSGI Server)

**Data & ML Stack**

* Python
* Pandas
* NumPy
* Scikit-learn
* LightGBM
* FAISS-CPU
* Torch (CPU version)
* Transformers == 4.30.2
* Sentence-Transformers == 2.2.2
* pickle-mixin (model serialization)

**MLOps & Deployment**

* Docker (Containerization)
* AWS ECS (Cloud Deployment)

---

## 🚀 API Endpoints (Example)

### Get Recommendations

```http
POST /recommend
```

**Request Body:**

```json
{
  "product_name": nail paints,
  "top_k": 5
}
```

**Response:**

```json
{
  "recommendations": [101, 205, 309, 412, 518]
}
```

*(Modify according to your actual implementation.)*

---

## 🐳 Running Locally with Docker

### Build the image

```bash
docker build -t product-recommendation-app .
```

### Run the container

```bash
docker run -p 8000:8000 product-recommendation-app
```

Access the API at:

```
http://localhost:8000
```

---

## ☁️ Deployment on AWS ECS

1. Build Docker image locally.
2. Push image to **AWS ECR**.
3. Create an **ECS Task Definition** using the image.
4. Launch an **ECS Service** behind an Application Load Balancer.
5. Monitor logs via **AWS CloudWatch**.

---

## 📊 Business Impact

This system helps businesses to:

* Improve product discovery
* Increase user engagement
* Boost conversion rates
* Provide AI-driven personalization
* Scale recommendations to millions of users

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit pull requests with clear documentation.

---

## 📬 Contact

For issues or collaboration, open a GitHub issue or connect via LinkedIn.

⭐ If this project helps you, please star the repository!
