import cloudinary.uploader
from .cloudinary_client import cloudinary

def upload_audio_to_cloudinary(file_bytes, filename):
    """
    Upload audio file bytes to Cloudinary.
    Returns the secure URL.
    """
    result = cloudinary.uploader.upload(
        file_bytes,
        resource_type="video",  # REQUIRED for audio uploads
        folder="speaker_audio/",
        public_id=filename.split('.')[0],  # remove extension
    )

    return result["secure_url"]
