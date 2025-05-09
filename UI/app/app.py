from flask import Flask, render_template, request
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_experimental.llms.ollama_functions import OllamaFunctions

app = Flask(__name__)

# Load vectorstore and retriever
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory="../vectorstore/chroma_db", embedding_function=embedding_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Load LLM
llm = OllamaFunctions(model="llama3:instruct", temperature=0.3, format="json")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Home page
@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    explanation = ""
    prompt=""

    if request.method == "POST":
        question = request.form.get("question")
        code = request.form.get("code")

        if question:
            result = qa_chain({"query": question})
            answer = result["result"]
            sources = "\n".join(f"- {doc.metadata['source']}" for doc in result["source_documents"])
            answer += "\n\n**Sources:**\n" + sources

        elif code:
            prompt = f"""
You are a helpful assistant trained on scikit-learn documentation.
Explain what this code does step-by-step and provide context.

```python
{code}
"""
    result = llm.invoke(prompt)
    explanation = result["content"] if isinstance(result, dict) else result

    return render_template("index.html", answer=answer, explanation=explanation)



app.run(debug=True)
