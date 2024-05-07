import sys
from pathlib import Path

from fastapi import FastAPI, File, UploadFile

# Adding model directory to system path to be able to import modules
sys.path.append(str(Path(__file__).parent.joinpath('model')))

from inference import infer

app = FastAPI()

@app.post("/predict/")
async def create_upload_file(file: UploadFile):
    # Save the uploaded file
    with open(f"images/test_image/{file.filename}", "wb") as f:
        f.write(file.file.read())
    
    # Run inference on the uploaded file
    image_path = f"images/test_image/{file.filename}"
    result = infer(image_path)
    
    # Return the inference result
    return {"filename": file.filename, "prediction": result}
