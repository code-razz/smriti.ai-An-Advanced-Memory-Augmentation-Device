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
