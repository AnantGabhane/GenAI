import json
from collections import Counter
from typing import List, Dict


class Encoder:
    def __init__(self):
        """init function"""
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}
        self.unk_token = "<UNK>"
        self.unk_id = 0
        self.space_token = "<SPACE>"
        self.space_id = 1

    def build_vocab(self, corpus: str, vocab_size: int, min_n: int = 2, max_n: int = 5):
        """vocab builder function"""
        corpus = corpus.replace(" ", "").lower()
        ngram_counter = Counter()

        # Count all n-grams in the given range
        for n in range(min_n, max_n + 1):
            for i in range(len(corpus) - n + 1):
                ngram = corpus[i : i + n]
                ngram_counter[ngram] += 1

        # Select top n-grams based on frequency
        most_common = ngram_counter.most_common(
            vocab_size - 2
        )  # Reserve 0 for <UNK>, 1 for <SPACE>

        self.token_to_id = {
            self.unk_token: self.unk_id,
            self.space_token: self.space_id,
        }
        self.id_to_token = {
            self.unk_id: self.unk_token,
            self.space_id: self.space_token,
        }

        for idx, (token, _) in enumerate(most_common, start=2):  # Start indexing from 2
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, text: str) -> List[int]:
        """encoder functioun - turns text into numbers"""
        text = text.lower()
        tokens = []
        i = 0
        max_token_length = max(map(len, self.token_to_id.keys()))

        while i < len(text):
            if text[i] == " ":
                tokens.append(self.space_id)
                i += 1
                continue

            matched = False
            for j in range(min(len(text), i + max_token_length), i, -1):
                substring = text[i:j]
                if substring in self.token_to_id:
                    tokens.append(self.token_to_id[substring])
                    i = j
                    matched = True
                    break
            if not matched:
                tokens.append(self.unk_id)
                i += 1
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """decoder function - turns numbers into text"""
        return "".join(
            [
                (
                    " "
                    if tid == self.space_id
                    else self.id_to_token.get(tid, self.unk_token)
                )
                for tid in token_ids
            ]
        )

    def save_vocab(self, path: str):
        """save vocab function"""
        with open(path, "w") as f:
            json.dump(self.token_to_id, f, indent=2)

    def load_vocab(self, path: str):
        """load vocab function"""
        with open(path, "r") as f:
            self.token_to_id = json.load(f)
            self.id_to_token = {v: k for k, v in self.token_to_id.items()}


if __name__ == "__main__":
    corpus = (
        "This is a sample English text to build a custom tokenizer from scratch."
        "My name is Anant and I'm a genai developer."
    )
    encoder = Encoder()
    encoder.build_vocab(corpus, vocab_size=100)

    test_text = "my name is anant and i'm a genai developer"
    encoded = encoder.encode(test_text)
    print("Encoded:", encoded)

    # Encoded: [85, 88, 1, 91, 94, 97, 1, 22, 61, 14]
    # Encoded: [57, 1, 6, 59, 1, 2, 1, 7, 7, 0, 1, 7, 0, 1, 65, 0, 1, 0, 1, 69, 6, 0, 1, 72, 74, 76, 78, 0]
    decoded = encoder.decode(encoded)
    print("Decoded:", decoded)
    # Decoded: custom tokenizer example
    # Decoded: my name is anan<UNK> an<UNK> i'<UNK> <UNK> gena<UNK> develope<UNK>
    encoder.save_vocab("vocab.json")
