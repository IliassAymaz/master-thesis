import torch
from transformers import BertTokenizer, BertModel

# MODEL = 'GTETE_backend/models/BERT_models/bert-bae-german-cased/'

# MODEL = 'GTETE_backend/models/BERT_models/bert-base-german-cased/'

MODEL = "bert-base-german-cased" # to be downloaded from the web!

class AfosEmbedding:
    def __init__(self, MODEL_):
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_)

        # load pre-trained german BERT model (weights)
        self.model = BertModel.from_pretrained(MODEL_, output_hidden_states=True)

        # put in evaluation mode
        self.model.eval()

    def sentence_embedding(self, sentence1, sentence2):
        """
        Feeds sentence1 and sentence2 to BERT and gets the two sentence embedding vectors.
        The last four layers are concatenated.
        The sentence vector is the mean of its tokens' vectors.

        params: sentence1, sentence2
        output: sentence_embeddings (list of two sentence vectors)
        """
        text = [sentence1, sentence2]
        encoded_text = []
        hidden_states_from_both_sentences = []
        for i in range(len(text)):
            # Feed each sentence into a separate model
            encoded_text.append(self.tokenizer.encode_plus(
                text[i],
                add_special_tokens=True
            ))
            tokens_tensor = torch.tensor([encoded_text[i]['input_ids']])
            segments_tensor = torch.tensor([encoded_text[i]['token_type_ids']])

            # model
            # Run the text through BERT, and collect all of the hidden states produced
            # from all 12 layers.
            with torch.no_grad():
                outputs = self.model(tokens_tensor, segments_tensor)
                hidden_states = outputs[2]
                hidden_states_from_both_sentences.append(hidden_states)

        sentence_embeddings = []
        for states in hidden_states_from_both_sentences:
            # Concatenate the last 4 layers
            # sentences_cat is now a tensor[len_sentence x 3072 (=768x4)]
            sentences_cat = torch.cat((states[-4][0], states[-3][0], states[-2][0], states[-1][0]), dim=1)

            # Get mean of the vectors to obtain the sentence embedding
            sentence_embedding = torch.mean(sentences_cat, dim=0)
            sentence_embeddings.append(sentence_embedding)
        return sentence_embeddings

# Example code:
# afos_embedding = AfosEmbedding(MODEL)
# list_of_sentence_embeddings = afos_embedding.sentence_embedding('GSI System', 'System Beschränkung')
# print(list_of_sentence_embeddings)
