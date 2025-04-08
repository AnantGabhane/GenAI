import tiktoken

encoder = tiktoken.encoding_for_model("gpt-4o")

print("Vocab size", encoder.n_vocab) # Vocab size 200019
# conclusion - gpt-4o has 200k vocab 

text = "The cat sat on the mat"
tokens = encoder.encode(text)

print("Tokens", tokens) # Tokens [976, 9059, 10139, 402, 290, 2450]

decoded = encoder.decode([976, 9059, 10139, 402, 290, 2450])
print("Decoded", decoded) # Decoded The cat sat on the mat
