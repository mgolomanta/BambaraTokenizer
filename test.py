from bambara_tokenizer.tokenizer import BambaraTokenizer

tokenizer = BambaraTokenizer()
text = "I bε angilekan men wa?" # Means Do you speak English?	
tokens = tokenizer.tokenize(text)
print(tokens)
