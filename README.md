# Face Recognition Backend API

A FastAPI-based backend service for face recognition, providing endpoints for face comparison, analysis, registration, and verification.

## Features

- Face comparison between two images
- Face analysis for detecting facial features
- Face registration and storage
- Face verification against registered faces
- List management of registered faces

## Prerequisites

- Python 3.8+
- OpenCV
- NumPy
- FastAPI
- uvicorn

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install fastapi uvicorn opencv-python numpy python-multipart
   ```

## Running the Server

Start the server using uvicorn:

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
- `GET /`
  - Returns API status

### Face Comparison
- `POST /compare-faces`
  - Compare two faces for similarity
  - Required: Two images as form data (`source_image` and `target_image`)

### Face Analysis
- `POST /analyze-face`
  - Analyze facial features in an image
  - Required: Image file as form data (`image`)

### Face Registration
- `POST /register-face/{face_id}`
  - Register a new face with specified ID
  - Required: 
    - `face_id`: Unique identifier for the face
    - Image file as form data (`image`)

### List Registered Faces
- `GET /registered-faces`
  - Get list of all registered faces

### Delete Face
- `DELETE /face/{face_id}`
  - Remove a registered face
  - Required: `face_id` of the face to delete

### Face Verification
- `POST /verify-face/{face_id}`
  - Verify a face against a registered face
  - Required:
    - `face_id`: ID of the registered face
    - Image file as form data (`image`)

## Error Handling

The API uses standard HTTP status codes:
- 200: Success
- 400: Bad Request
- 404: Not Found
- 500: Internal Server Error

Errors include descriptive messages in the response body.

## Security Considerations

- CORS is enabled with configurable origins
- Input validation for all endpoints
- Error handling for invalid requests

## Contributing

Contributions are welcome! Please feel free to submit pull requests.

## License

MIT License