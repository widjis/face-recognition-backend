import requests
import os
import json

# Configuration
BASE_URL = "http://localhost:8000"

def test_compare_faces():
    """Test the /compare-faces endpoint"""
    print("\nTesting /compare-faces endpoint...")
    
    # Test file paths (you should replace these with actual test image paths)
    test_image1_path = "test_images/MTI230277.jpg"
    test_image2_path = "test_images/MTI230279.jpg"
    
    # Ensure test files exist
    if not (os.path.exists(test_image1_path) and os.path.exists(test_image2_path)):
        print("Error: Test image files not found. Please add test images to test_images directory.")
        return False
    
    try:
        # Prepare the files for upload
        with open(test_image1_path, 'rb') as f1, open(test_image2_path, 'rb') as f2:
            files = {
                'source_image': ('face1.jpg', f1, 'image/jpeg'),
                'target_image': ('face2.jpg', f2, 'image/jpeg')
            }
            
            # Make the request
            response = requests.post(f"{BASE_URL}/compare-faces", files=files)
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                print("Success! Response:")
                print(json.dumps(result, indent=2))
                
                # Validate response structure
                assert 'verified' in result, "Response missing 'verified' field"
                assert 'distance' in result, "Response missing 'distance' field"
                assert 'threshold' in result, "Response missing 'threshold' field"
                
                print("✓ Response structure is valid")
                return True
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the API is running.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_analyze_face():
    """Test the /analyze-face endpoint"""
    print("\nTesting /analyze-face endpoint...")
    
    # Test file path
    test_image_path = "test_images/MTI230277.jpg"
    
    # Ensure test file exists
    if not os.path.exists(test_image_path):
        print("Error: Test image file not found. Please add test images to test_images directory.")
        return False
    
    try:
        # Prepare the file for upload
        with open(test_image_path, 'rb') as f:
            files = {
                'image': ('face.jpg', f, 'image/jpeg')
            }
            
            # Make the request
            response = requests.post(f"{BASE_URL}/analyze-face", files=files)
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                print("Success! Response:")
                print(json.dumps(result, indent=2))
                
                # Validate response structure contains facial analysis data
                assert isinstance(result, dict), "Response should be a dictionary"
                
                print("✓ Response structure is valid")
                return True
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the API is running.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_register_face():
    """Test the /register-face/{face_id} endpoint by registering all images in test_images directory"""
    print("\nTesting /register-face endpoint...")
    
    # Get all jpg files from test_images directory
    test_images_dir = "test_images"
    if not os.path.exists(test_images_dir):
        print(f"Error: {test_images_dir} directory not found.")
        return False
    
    image_files = [f for f in os.listdir(test_images_dir) if f.lower().endswith('.jpg')]
    if not image_files:
        print(f"Error: No jpg images found in {test_images_dir} directory.")
        return False
    
    success_count = 0
    total_files = len(image_files)
    
    print(f"Found {total_files} images to register...")
    
    for image_file in image_files:
        test_image_path = os.path.join(test_images_dir, image_file)
        # Use filename without extension as face_id
        face_id = os.path.splitext(image_file)[0]
        
        try:
            # Prepare the file for upload
            with open(test_image_path, 'rb') as f:
                files = {
                    'image': ('face.jpg', f, 'image/jpeg')
                }
                
                # Make the request
                response = requests.post(f"{BASE_URL}/register-face/{face_id}", files=files)
                
                # Check response status
                if response.status_code == 200:
                    result = response.json()
                    print(f"✓ Successfully registered {face_id}")
                    success_count += 1
                else:
                    print(f"✗ Failed to register {face_id}: {response.text}")
                    
        except requests.exceptions.ConnectionError:
            print(f"✗ Failed to register {face_id}: Could not connect to the server")
        except Exception as e:
            print(f"✗ Failed to register {face_id}: {str(e)}")
    
    print(f"\nRegistration Summary:")
    print(f"Successfully registered: {success_count}/{total_files} images")
    
    return success_count > 0

def test_verify_face():
    """Test the /verify-face/{face_id} endpoint"""
    print("\nTesting /verify-face endpoint...")
    
    # Test file path and face ID
    test_image_path = "test_images/MTI230277.jpg"
    test_face_id = "test_face_1"
    
    # Ensure test file exists
    if not os.path.exists(test_image_path):
        print("Error: Test image file not found. Please add test images to test_images directory.")
        return False
    
    try:
        # First register a face to verify against
        with open(test_image_path, 'rb') as f:
            files = {
                'image': ('face.jpg', f, 'image/jpeg')
            }
            response = requests.post(f"{BASE_URL}/register-face/{test_face_id}", files=files)
            
            if response.status_code != 200:
                print("Error: Failed to register test face")
                return False
        
        # Now verify the same face (in a real scenario, you'd use a different image)
        with open(test_image_path, 'rb') as f:
            files = {
                'image': ('face.jpg', f, 'image/jpeg')
            }
            
            # Make the verification request
            response = requests.post(f"{BASE_URL}/verify-face/{test_face_id}", files=files)
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                print("Success! Response:")
                print(json.dumps(result, indent=2))
                
                # Validate response structure
                assert 'verified' in result, "Response missing 'verified' field"
                assert 'distance' in result, "Response missing 'distance' field"
                assert 'threshold' in result, "Response missing 'threshold' field"
                
                print("✓ Response structure is valid")
                return True
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the API is running.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def test_search_face():
    """Test the /search-face endpoint"""
    print("\nTesting /search-face endpoint...")
    
    # Test file path (using a known good test image)
    test_image_path = "test_images/tes.jpg"
    
    # Ensure test file exists
    if not os.path.exists(test_image_path):
        print("Error: Test image file not found. Please add test images to test_images directory.")
        return False
    
    try:
        # Prepare the file for upload
        with open(test_image_path, 'rb') as f:
            files = {
                'image': ('face.jpg', f, 'image/jpeg')
            }
            
            # Make the request
            response = requests.post(f"{BASE_URL}/search-face", files=files)
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                print("Success! Response:")
                print(json.dumps(result, indent=2))
                
                # Validate response structure
                assert isinstance(result, list), "Response should be a list of matches"
                if len(result) > 0:
                    assert 'face_id' in result[0], "Response missing 'face_id' field"
                    assert 'distance' in result[0], "Response missing 'distance' field"
                
                print("✓ Response structure is valid")
                return True
            else:
                print(f"Error: Received status code {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the API is running.")
        return False
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Run all API tests"""
    print("Starting API tests...")
    
    # Create test_images directory if it doesn't exist
    os.makedirs("test_images", exist_ok=True)
    
    # Test health check endpoint
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            print("✓ API is running")
        else:
            print("✗ API is not responding correctly")
            return
    except requests.exceptions.ConnectionError:
        print("✗ Could not connect to the API. Make sure the server is running.")
        return
    
    # Run tests
    tests = [
        #("Compare Faces", test_compare_faces),
        #("Register Face", test_register_face),
        #("Analyze Face", test_analyze_face),
        #("Verify Face", test_verify_face),
        ("Search Face", test_search_face),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning test: {test_name}")
        success = test_func()
        results.append((test_name, success))
    
    # Print summary
    print("\nTest Summary:")
    for test_name, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")

if __name__ == "__main__":
    main()