from transformers import AutoModel, AutoTokenizer
import pickle

text_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

train_notes = np.load("train_notes.npy", allow_pickle=True)
test_notes = np.load("test_notes.npy", allow_pickle=True)
train_ids = np.load("train_x_idx.npy", allow_pickle=True)
test_ids = np.load("test_x_idx.npy", allow_pickle=True)

def get_text_embeddings(clinical_notes, seq_len=24):
    encoded_notes = []
    accumulation_time = 0
    six_hour_chunks = int(seq_len/6)
    CHUNK_LEN = 512
    encoded_notes = []
    for i in range(six_hour_chunks):
        encoded_notes.append(tokenizer.encode("", return_tensors='pt'))

    for note in clinical_notes:
        note_time = note[0]
        note_text = note[1]
        note_quarter = int(note_time/6)
        # Quarters for which notes are available
        if note_quarter <= six_hour_chunks:
            note_text_chunks = [note_text[i:i+CHUNK_LEN] for i in range(0, len(note_text), CHUNK_LEN)]
            for chunk in note_text_chunks:
                encoded_notes.append(tokenizer.encode(chunk, return_tensors='pt'))
        else:
            break

    # Define the loss function and optimizer
    note_embeddings = [text_model(note)[0] for note in encoded_notes]
    average_embeddings = [embedding.mean(dim=1) for embedding in note_embeddings]
    final_vec = torch.zeros(average_embeddings[0].shape)
    for sentence_embed in average_embeddings:
        final_vec = 0.7*final_vec + 0.3*sentence_embed
    
    return final_vec

embeddings = {}
for i in tqdm(range(len(train_notes))):
    pat_key = train_notes[i][0]
    emb = get_text_embeddings(train_notes[i][1])
    embeddings[pat_key] = emb

with open('embeddings.pickle', 'wb') as handle:
    pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
