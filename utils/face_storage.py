import os
import pickle
import numpy as np
from pathlib import Path
from deepface import DeepFace
from typing import Dict, List, Optional

class FaceStorage:
    def __init__(self, storage_dir: str = 'face_database'):
        self.storage_dir = Path(storage_dir)
        self.embeddings_dir = self.storage_dir / 'embeddings'
        self.images_dir = self.storage_dir / 'images'
        self.embeddings_file = self.embeddings_dir / 'face_embeddings.pkl'
        self.face_embeddings = {}
        self._initialize_storage()
        self._load_embeddings()

    def _initialize_storage(self):
        """Create necessary directories if they don't exist and initialize embeddings storage."""
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if not self.embeddings_file.exists():
            self._save_embeddings()

    def _load_embeddings(self):
        """Load face embeddings from the single storage file."""
        try:
            if self.embeddings_file.exists():
                with open(self.embeddings_file, 'rb') as f:
                    self.face_embeddings = pickle.load(f)
        except Exception as e:
            print(f'Warning: Failed to load embeddings: {str(e)}')
            self.face_embeddings = {}

    def _save_embeddings(self):
        """Save all face embeddings to the single storage file."""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.face_embeddings, f)
        except Exception as e:
            raise ValueError(f'Failed to save embeddings: {str(e)}')

    def extract_embedding(self, image_array) -> np.ndarray:
        """Extract face embedding from an image array using DeepFace."""
        try:
            embedding = DeepFace.represent(
                img_path=image_array,
                model_name='ArcFace',
                enforce_detection=True,
                detector_backend='opencv'
            )
            return embedding[0]['embedding']
        except Exception as e:
            raise ValueError(f'Failed to extract face embedding: {str(e)}')

    def save_face(self, face_id: str, image_array, embedding: Optional[np.ndarray] = None) -> bool:
        """Save face embedding to storage."""
        try:
            if embedding is None:
                embedding = self.extract_embedding(image_array)

            # Save embedding to dictionary and persist to file
            self.face_embeddings[face_id] = embedding
            self._save_embeddings()

            return True
        except Exception as e:
            raise ValueError(f'Failed to save face: {str(e)}')

    def get_face_embedding(self, face_id: str) -> Optional[np.ndarray]:
        """Retrieve face embedding by ID."""
        return self.face_embeddings.get(face_id)

    def list_registered_faces(self) -> List[str]:
        """Get a list of all registered face IDs."""
        return list(self.face_embeddings.keys())

    def delete_face(self, face_id: str) -> bool:
        """Delete a face from storage."""
        try:
            if face_id in self.face_embeddings:
                del self.face_embeddings[face_id]
                self._save_embeddings()
            return True
        except Exception:
            return False

    def verify_face(self, image_array, face_id: str) -> Dict:
        """Verify if a given face matches with a registered face ID."""
        try:
            # Get stored embedding
            stored_embedding = self.get_face_embedding(face_id)
            if stored_embedding is None:
                raise ValueError(f'No face found with ID: {face_id}')

            # Get new face embedding
            new_embedding = self.extract_embedding(image_array)

            # Calculate similarity
            distance = np.linalg.norm(new_embedding - stored_embedding)
            threshold = 0.6  # ArcFace threshold
            verified = distance <= threshold

            return {
                'verified': verified,
                'distance': float(distance),
                'threshold': threshold
            }
        except Exception as e:
            raise ValueError(f'Face verification failed: {str(e)}')