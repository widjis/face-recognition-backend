from deepface import DeepFace
import numpy as np

class FaceRecognition:
    @staticmethod
    def compare_faces(source_img, target_img):
        """Compare two face images and return verification result."""
        try:
            result = DeepFace.verify(
                img1_path=source_img,
                img2_path=target_img,
                enforce_detection=True,
                model_name='ArcFace'
            )

            return {
                "verified": result["verified"],
                "distance": float(result["distance"]),
                "threshold": float(result["threshold"])
            }
        except Exception as e:
            raise ValueError(f"Face comparison failed: {str(e)}")

    @staticmethod
    def analyze_face(image):
        """Analyze a face image for age, gender, emotion, and race."""
        try:
            result = DeepFace.analyze(
                img_path=image,
                actions=['age', 'gender', 'emotion', 'race'],
                enforce_detection=True
            )

            return result[0] if isinstance(result, list) else result
        except Exception as e:
            raise ValueError(f"Face analysis failed: {str(e)}")

    @staticmethod
    def extract_faces(image):
        """Extract faces from an image using DeepFace.

        Args:
            image: Image file path or numpy array.

        Returns:
            List of detected faces as numpy arrays.
        """
        try:
            result = DeepFace.extract_faces(
                img_path=image,
                target_size=(224, 224),
                detector_backend='opencv',
                enforce_detection=True
            )
            
            return [face['face'] for face in result]
        except Exception as e:
            raise ValueError(f"Face extraction failed: {str(e)}")