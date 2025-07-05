import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np
import os

import torch
from transformers import BertTokenizer, BertModel, AlbertTokenizer, AlbertModel, XLNetTokenizer, XLNetModel, \
    DistilBertModel, DistilBertTokenizer

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", Warning)  # Ignore all warnings

class FakeEmbedding:

    def __init__(
            self,
            embedding_size,
    ):
        # TBD Change to local env
        self.home_dir = "/Users/dpeleg"
        self.download_nltk_data()
        # self.lemmatizer = WordNetLemmatizer()
        self.df = None

        # Load the LIAR dataset
        # TBD Change to local env. Download liar dataset
        self.root_dir = f"{self.home_dir}/local/fake_dataset/liar"
        self.file_name = "train2.tsv"
        # self.file_name = "test2.tsv"
        # self.file_name = "dev.tsv"

        # Load pre-trained BERT model and tokenizer
        # self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self.model = BertModel.from_pretrained('bert-base-uncased')

        if (embedding_size == 'albert-xxlarge-v2'):
            self.tokenizer = AlbertTokenizer.from_pretrained('albert-xxlarge-v2')
            self.model = AlbertModel.from_pretrained('albert-xxlarge-v2')
        elif (embedding_size == 'xlnet-large-cased'):
            self.tokenizer = XLNetTokenizer.from_pretrained('xlnet-large-cased')
            self.model = XLNetModel.from_pretrained('xlnet-large-cased')
        elif (embedding_size == 'distilbert-base-uncased'):
            self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        elif (embedding_size == 'bert-base-uncased'):
            self.tokenizer = BertTokenizer.from_pretrained(embedding_size)
            self.model = BertModel.from_pretrained(embedding_size)


    def download_nltk_data(self):
        nltk_checkpoint_path = f"{self.home_dir}/nltk_data"
        if not os.path.exists(nltk_checkpoint_path):
            print("Downloading NLTK dataset")
            nltk.download('punkt')
            nltk.download('punkt_tab')
            nltk.download('stopwords')
            nltk.download('wordnet')

            with open(nltk_checkpoint_path, 'w') as f:
                pass  # Create an empty file
            print(f"File {nltk_checkpoint_path} created.")
        else:
            print("nltk data already downloaded")


    def load_dataset(self):
        # tokenizer = word_tokenize('hello world', language='english')

        # Define the column names
        column_names = ['rec_id', 'id', 'label', 'statement', 'subject', 'speaker', 'job title', 'state', 'party',
                        'true', 'false', 'half', 'mostly', 'pants-fire', 'context', 'justification']

        self.df = pd.read_csv(f"{self.root_dir}/{self.file_name}", delimiter='\t', names=column_names)

        # Describe the dataset
        print(self.df.describe())
        # Print the top 3 lines of the dataset
        print(self.df.columns)
        print(self.df.head(3))

        X, y = self.preprocess()

        return X, y

    # # Define a function to preprocess the text data
    def preprocess_text(self, text):
        if isinstance(text, str):  # Check if text is a string
            # Tokenize the text
            tokens = word_tokenize(text)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            tokens = [token for token in tokens if token.lower() not in stop_words]

            # Lemmatize the tokens
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

            # Join the tokens back into a string
            text = ' '.join(tokens)

            return text
        else:
            return ''  # Return an empty string for non-string inputs

    def preprocess(self):
        # Apply the preprocessing function to the statement column
        self.df['statement'] = self.df['statement'].apply(self.preprocess_text)

        # Extract the article texts
        statement = self.df['statement'].fillna('')

        # Generate BERT embeddings for the article texts
        embeddings = self.generate_bert_embeddings(statement)

        # Convert the embeddings to a numpy array
        embeddings_array = np.array(embeddings)
        # remove dimension for 2D
        embeddings_array = embeddings_array.squeeze()

        # y = np.random.randint(0, 5, size=100)
        y = self.df['label'].values

        return embeddings_array, y

    def generate_bert_embeddings(self, texts):
        # Create a list to store the embeddings
        embeddings = []

        # Iterate over the input texts
        i = 0
        for text in texts:
            i += 1
            if i % 1000 == 0:
                print(f"Completed {i} loops in method")
            # Tokenize the text using the BERT tokenizer
            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=512,
                return_attention_mask=True,
                return_tensors='pt',
                truncation=True
            )

            # Get the input IDs and attention mask
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']

            # Zero the gradients
            self.model.zero_grad()

            # Forward pass through the BERT model
            outputs = self.model(input_ids, attention_mask=attention_mask)

            # Get the last hidden state (i.e., the embedding)
            embedding = outputs.last_hidden_state[:, 0, :]
            # print(embedding.shape)  # Output: torch.Size([1, 768])

            # Append the embedding to the list
            embeddings.append(embedding.detach().numpy())

        print(f"Completed {i} loops in function")

        # Return the list of embeddings
        return embeddings
