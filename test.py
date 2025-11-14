from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2")
A = "are you free today?"
B = "flower is beautiful"
emb_a = model.encode(A, convert_to_tensor=True)
emb_b = model.encode(B, convert_to_tensor=True)
cos = util.cos_sim(emb_a, emb_b).item()
print("cosine:", cos)              # ~0.03-0.06 typical here
print("percent:", cos*100)
print("norm A:", float(emb_a.norm()))
print("norm B:", float(emb_b.norm()))
