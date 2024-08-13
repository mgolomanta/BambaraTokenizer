from bambara_tokenizer.tokenizer import BambaraTokenizer

tokenizer = BambaraTokenizer()
text = "sample Bambara text"
tokens = tokenizer.tokenize(text)
print(tokens)