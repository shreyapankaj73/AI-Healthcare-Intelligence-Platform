import json
import ollama
import fitz
import tempfile
import os


def pdf_to_image(pdf_path):

    doc = fitz.open(pdf_path)

    page = doc[0]

    pix = page.get_pixmap(
        matrix=fitz.Matrix(2, 2)
    )

    output_path = os.path.join(
        tempfile.gettempdir(),
        "medical_report_page1.png"
    )

    if os.path.exists(output_path):
        try:
            os.remove(output_path)
        except:
            pass

    pix.save(output_path)

    doc.close()

    return output_path


def extract_medical_data(file_path):

    image_path = file_path

    if file_path.lower().endswith(".pdf"):
        image_path = pdf_to_image(file_path)

        print("Image path:", image_path)
    print("Exists:", os.path.exists(image_path))


    prompt = """
You are an expert medical report analyst.

Extract ALL medical values visible in the report.

Return ONLY valid JSON.

{
    "name":"",
    "age":null,
    "gender":"",
    "mobile":"",
    "uhid":"",

    "glucose":null,
    "hba1c":null,

    "hemoglobin":null,
    "rbc":null,
    "wbc":null,
    "platelets":null,

    "hematocrit":null,
    "mcv":null,
    "mch":null,
    "mchc":null,
    "rdw":null,

    "cholesterol":null,
    "hdl":null,
    "ldl":null,
    "triglycerides":null,

    "creatinine":null,
    "urea":null,
    "uric_acid":null,

    "bilirubin":null,
    "sgot":null,
    "sgpt":null,

    "sodium":null,
    "potassium":null,

    "oxygen":null,
    "heart_rate":null,

    "abnormalities":[],
    "recommendations":[],

    "summary":"Provide a detailed 10 sentence medical interpretation."
}
"""



    response = ollama.chat(
        model="gemma3",
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_path]
            }
        ]
    )

    content = response["message"]["content"]

    start = content.find("{")
    end = content.rfind("}")

    if start == -1 or end == -1:
        raise Exception("No JSON returned by model")

    return json.loads(content[start:end + 1])