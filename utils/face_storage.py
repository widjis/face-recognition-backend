import os
import pickle
import cv2
from pathlib import Path

class FaceStorage:
    def __init__(self):
        self.base_dir = Path('face_database')
        self.images_dir = self.base_dir / 'images'
        self.embeddings_dir = self.base_dir / 'embeddings'
        
        # Create directories if they don't exist
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def save_face(self, face_id: str, image):
        """Save a face image and its embedding"""
        # Save the image
        image_path = self.images_dir / f"{face_id}.jpg"
        cv2.imwrite(str(image_path), image)

        # Save the embedding
        from services.face_recognition import FaceRecognition
        embedding = FaceRecognition.get_face_embedding(image)
        embedding_path = self.embeddings_dir / f"{face_id}.pkl"
        with open(embedding_path, 'wb') as f:
            pickle.dump(embedding, f)

    def get_face(self, face_id: str):
        """Get a face image and its embedding"""
        image_path = self.images_dir / f"{face_id}.jpg"
        embedding_path = self.embeddings_dir / f"{face_id}.pkl"

        if not image_path.exists() or not embedding_path.exists():
            return None

        image = cv2.imread(str(image_path))
        with open(embedding_path, 'rb') as f:
            embedding = pickle.load(f)

        return {'image': image, 'embedding': embedding}

    def list_registered_faces(self):
        """List all registered face IDs"""
        try:
            # Get all .jpg files in the images directory
            face_files = list(self.images_dir.glob('*.jpg'))
            # Extract face IDs (filename without extension)
            face_ids = [f.stem for f in face_files]
            return face_ids
        except Exception as e:
            print(f"Error listing faces: {str(e)}")
            return []

    def delete_face(self, face_id: str) -> bool:
        """Delete a face's image and embedding"""
        image_path = self.images_dir / f"{face_id}.jpg"
        embedding_path = self.embeddings_dir / f"{face_id}.pkl"

        try:
            if image_path.exists():
                image_path.unlink()
            if embedding_path.exists():
                embedding_path.unlink()
            return True
        except Exception as e:
            print(f"Error deleting face {face_id}: {str(e)}")
            return False