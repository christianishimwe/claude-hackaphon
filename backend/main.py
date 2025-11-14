# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import process_rules_pdf, generate_apology

app = FastAPI()

# Allow local React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for hackathon, keep open
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApologyRequest(BaseModel):
    caseDescription: str
    wrongdoing: str


@app.post("/upload-rules")
async def upload_rules(file: UploadFile = File(...)):
    """
    Upload the PDF containing all apology rules/cases.
    """
    count = process_rules_pdf(file)
    return {
        "message": "Rules uploaded and indexed successfully.",
        "casesIndexed": count,
    }


@app.post("/generate-apology")
async def create_apology(req: ApologyRequest):
    apology = generate_apology(
        case_description=req.caseDescription,
        wrongdoing=req.wrongdoing,
    )
    return {"apology": apology}
