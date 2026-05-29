Corpus_data = "loreum ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."

import os

from google import genai
client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

result=client.models.embed_content(model="models/gemini-embedding-001", contents=Corpus_data)


print(result)

# print("=============================")
# print(result.embeddings)

# print("=============================")

print(result.embeddings[0].values) # type: ignore
print("Vector length: ", len(result.embeddings[0].values)) # type: ignore

# Note a vector db is sentence to vector and not word to vector, so the embedding is for the whole sentence and not for each word in the sentence.
# Note the length of vector db never changes even if u add more word to the sentence.
