import os
import json

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI

load_dotenv()

app = Flask(__name__)

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is missing from your .env file")

client = OpenAI(api_key=api_key)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():

    if "report" not in request.files:
        return jsonify({"error": "No report uploaded"}), 400

    file = request.files["report"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not file.filename.lower().endswith(".pdf"):
        return jsonify({
            "error": "Please upload a PDF file."
        }), 400

    try:
        # Read PDF
        reader = PdfReader(file)

        report_text = ""

        for page in reader.pages:
            report_text += page.extract_text() or ""

        if not report_text.strip():
            return jsonify({
                "error": "Could not read text from this PDF."
            }), 400

        # Instructions for the AI
        instructions = """
You are an AI assistant that summarizes medical reports.

Analyze ONLY the information contained in the supplied report.

Return ONLY valid JSON using this structure:

{
    "summary": "Simple explanation of the report",
    "medications": [
        {
            "name": "Medication name",
            "purpose": "Purpose if stated in the report"
        }
    ],
    "conditions": [
        "Conditions explicitly mentioned in the report"
    ],
    "abnormal_results": [
        {
            "test": "Test name",
            "value": "Reported value",
            "reference_range": "Reference range if available",
            "interpretation": "High, low, normal, or flagged according to the report"
        }
    ],
    "follow_up": [
        "Things the patient should discuss with a qualified healthcare professional"
    ],
    "missing_information": [
        "Important information that cannot be determined from this report"
    ]
}

Safety requirements:

- Do not diagnose conditions that are not established in the report.
- Do not claim that the patient is completely healthy.
- Do not recommend starting, stopping, or changing medication.
- Do not invent test results.
- If information is unclear, say it is unclear.
- Keep the explanation understandable to a general reader.
"""

        prompt = instructions + "\n\nMEDICAL REPORT:\n" + report_text

        # Send report to OpenAI
        response = client.responses.create(
            model="gpt-6-luna",
            input=prompt
        )

        result_text = response.output_text.strip()

        # Remove markdown formatting if returned
        if result_text.startswith("```"):
            result_text = result_text.replace("```json", "")
            result_text = result_text.replace("```", "")
            result_text = result_text.strip()

        result = json.loads(result_text)

        return jsonify(result)

    except json.JSONDecodeError:
        return jsonify({
            "error": "The AI returned an unexpected response format."
        }), 500

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":
    app.run(debug=True)