from flask import Flask, request, jsonify
import requests
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

API_URL = (
    "https://api-inference.huggingface.co/models/pszemraj/pegasus-x-large-book-summary"
)
headers = {"Authorization": "Bearer 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"}


# Function to query the Hugging Face API and get the summary for a given text
def query_api(payload):
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result[0]["summary_text"]
    except requests.exceptions.RequestException as e:
        print("Error querying the API:", e)
        return None


# Function to summarize pages in a batch
def summarize_pages_batch(pdf_reader, start_page, end_page):
    pages = pdf_reader.pages[start_page:end_page]
    text = "".join([page.extract_text() for page in pages])
    return query_api({"inputs": text})


def extract_text_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text_array = [page.extract_text() for page in pdf_reader.pages]
    return pdf_reader, text_array


@app.route("/summarize-pdf", methods=["POST"])
def summarize_pdf():
    if "pdf_file" not in request.files:
        return jsonify({"error": "No PDF file uploaded"}), 400

    pdf_file = request.files["pdf_file"]
    if pdf_file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if pdf_file and pdf_file.filename.endswith(".pdf"):
        try:
            pdf_reader, text_array = extract_text_from_pdf(pdf_file)
            num_pages = len(text_array)

            batch_size = num_pages // 5  # Number of pages to summarize in each batch
            output = []

            with ThreadPoolExecutor() as executor:
                for i in range(0, num_pages, batch_size):
                    end_page = min(i + batch_size, num_pages)
                    future = executor.submit(
                        summarize_pages_batch, pdf_reader, i, end_page
                    )
                    output.append(future)

            summaries = [
                future.result() for future in output if future.result() is not None
            ]
            joined_summary = " ".join(summaries)

            return jsonify({"result": joined_summary}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    else:
        return (
            jsonify({"error": "Invalid file format. Please provide a PDF file."}),
            400,
        )


if __name__ == "__main__":
    app.run(debug=True)
