from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import numpy as np
import cv2
import io
from utils.face_storage import FaceStorage
from services.face_recognition import FaceRecognition
from fastapi.middleware.trustedhost import TrustedHostMiddleware

# Initialize FastAPI app first
app = FastAPI()

# Initialize face storage
face_storage = FaceStorage()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Face Recognition API is running 123"}

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

        if img is None:
            raise ValueError("Failed to decode image")

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

@app.post("/search-face")
async def search_face(image: UploadFile = File(...)):
    try:
        # Read and convert query image
        content = await image.read()
        np_arr = np.frombuffer(content, np.uint8)
        query_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if query_img is None:
            raise ValueError("Failed to decode image")

        # Get query face embedding
        query_embedding = FaceRecognition.get_face_embedding(query_img)

        # Get all registered faces
        registered_faces = face_storage.list_registered_faces()
        print(f"Found {len(registered_faces)} registered faces: {registered_faces}")
        matches = []

        # Compare query face embedding with all registered faces
        for face_id in registered_faces:
            registered_face = face_storage.get_face(face_id)
            if registered_face:
                try:
                    # Get pre-computed embedding for registered face
                    registered_embedding = registered_face['embedding']
                    
                    # Calculate cosine similarity
                    similarity = float(np.dot(query_embedding, registered_embedding) / \
                               (np.linalg.norm(query_embedding) * np.linalg.norm(registered_embedding)))
                    
                    # Convert similarity to distance (lower is better)
                    distance = float(1 - similarity)
                    
                    # Use same threshold as in compare_faces
                    verified = bool(distance < 0.6)  # Convert numpy.bool_ to Python bool
                    
                    matches.append({
                        "face_id": face_id,
                        "distance": distance,
                        "verified": verified
                    })
                except Exception as e:
                    print(f"Failed to compare with face {face_id}: {str(e)}")
                    continue

        # Sort matches by distance (lower is better)
        matches.sort(key=lambda x: x["distance"])
        return matches

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/verify-face/{face_id}")
async def verify_face(face_id: str, image: UploadFile = File(...)):
    try:
        # Read and convert query image
        content = await image.read()
        np_arr = np.frombuffer(content, np.uint8)
        query_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Get registered face
        registered_face = face_storage.get_face(face_id)
        if not registered_face:
            raise HTTPException(status_code=404, detail=f"Face with ID {face_id} not found")

        # Compare faces
        result = FaceRecognition.compare_faces(query_img, registered_face['image'])
        return result

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)