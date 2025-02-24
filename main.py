from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import io
from utils.face_storage import FaceStorage
from services.face_recognition import FaceRecognition

# Initialize face storage
face_storage = FaceStorage()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running"}

@app.post("/compare-faces")
async def compare_faces(source_image: UploadFile = File(...), target_image: UploadFile = File(...)):
    try:
        # Read and convert source image
        source_content = await source_image.read()
        source_np = np.frombuffer(source_content, np.uint8)
        source_img = cv2.imdecode(source_np, cv2.IMREAD_COLOR)

        # Read and convert target image
        target_content = await target_image.read()
        target_np = np.frombuffer(target_content, np.uint8)
        target_img = cv2.imdecode(target_np, cv2.IMREAD_COLOR)

        # Perform face verification
        result = FaceRecognition.compare_faces(source_img, target_img)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/analyze-face")
async def analyze_face(image: UploadFile = File(...)):
    try:
        # Read and convert image
        content = await image.read()
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Analyze face
        result = FaceRecognition.analyze_face(img)
        return result

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/register-face/{face_id}")
async def register_face(face_id: str, image: UploadFile = File(...)):
    try:
        # Read and convert image
        content = await image.read()
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Register face
        face_storage.save_face(face_id, img)
        return {"message": f"Face registered successfully with ID: {face_id}"}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/registered-faces")
async def list_faces():
    try:
        faces = face_storage.list_registered_faces()
        return {"registered_faces": faces}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/face/{face_id}")
async def delete_face(face_id: str):
    try:
        if face_storage.delete_face(face_id):
            return {"message": f"Face with ID {face_id} deleted successfully"}
        raise HTTPException(status_code=404, detail=f"Face with ID {face_id} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/verify-face/{face_id}")
async def verify_face(face_id: str, image: UploadFile = File(...)):
    try:
        # Read and convert image
        content = await image.read()
        np_arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Verify face
        result = face_storage.verify_face(img, face_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))