import cloudinary
import cloudinary.uploader
import logging
import os
from config import CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET

# Configure Cloudinary
cloudinary.config(
    cloud_name=CLOUDINARY_CLOUD_NAME,
    api_key=CLOUDINARY_API_KEY,
    api_secret=CLOUDINARY_API_SECRET
)

def upload_audio_to_cloudinary(file_path, public_id):
    """
    Upload an audio file to Cloudinary.

    Parameters:
        file_path (str or Path): Path to the audio file.
        public_id (str): The public ID to assign to the uploaded file.

    Returns:
        str: The secure URL of the uploaded audio, or None if failed.
    """
    try:
        logging.info(f"☁️ Uploading {file_path} to Cloudinary as {public_id}...")
        response = cloudinary.uploader.upload(
            str(file_path),
            resource_type="video",  # 'video' is used for audio in Cloudinary
            public_id=public_id,
            folder="unknown_speakers"
        )
        url = response.get("secure_url")
        logging.info(f"✅ Cloudinary upload successful: {url}")
        return url
    except Exception as e:
        logging.error(f"❌ Cloudinary upload failed: {e}")
        return None

def upload_face_image(image_bytes: bytes, person_name: str) -> dict:
    """
    Uploads a face image (JPEG bytes) to Cloudinary.
    - image_bytes: cropped face (JPEG bytes)
    - person_name: label/name

    Returns:
      {
        "public_id": "...",
        "url": "http://...",
        "secure_url": "https://..."
      }
    """
    import time
    timestamp = int(time.time())
    # e.g. smriti/faces/Atia_1733691234
    public_id_base = f"smriti/faces/{person_name}_{timestamp}"

    try:
        logging.info(f"☁️ Uploading face for {person_name} to Cloudinary...")
        result = cloudinary.uploader.upload(
            image_bytes,
            public_id=public_id_base,
            folder="smriti/faces",
            resource_type="image",
            tags=["ai-smriti", f"person:{person_name}"],
            context={"name": person_name},
            overwrite=False,
        )
        
        logging.info(f"✅ Cloudinary face upload successful: {result.get('secure_url')}")
        return {
            "public_id": result.get("public_id"),
            "url": result.get("url"),
            "secure_url": result.get("secure_url"),
        }
    except Exception as e:
        logging.error(f"❌ Cloudinary face upload failed: {e}")
        return {}
