from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_sent_embedding( sent ):
    return model.encode( sent )