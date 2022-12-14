from sentence_transformers import SentenceTransformer

model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

def get_sent_embedding( sent ):
    return model.encode( sent , convert_to_tensor=True , convert_to_numpy=False )
