from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from typing import Annotated

from vision import detector, identifier
from vision.identifier import bytes_to_image


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup models for detection and more
    identifier.load_models()
    detector.load_models()

    yield

    # Clean up the ML models and release the resources
    identifier.cleanup_models()
    detector.cleanup_models()


app = FastAPI()


ALLOWED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"]

@app.post("/uploadbrick/")
async def upload_brick(file: UploadFile):
    """
    Whenever a brick is uploaded, it will be compared against
    the bricks currently saved in the system.
    If there are any matches, the user gets a session token
    that serves as a temporary master password they can use
    to access their passwords.

    :param file:    The file that was uploaded
                    (may or may not be image/brick)

    :return:        BAD REQUEST - Not an image <br>
                    NOT FOUND - Not even a brick smh my head <br>
                    UNAUTHORIZED - Was a brick, but bad brick <br>
                    INTERNAL SERVER ERROR - Could not convert to CV image <br>
                    OK - Signed in! Attached is a session token
    """
    filename = file.filename
    content_type = file.content_type

    # Validate image
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code = 404,
            detail = f"File type {content_type} is not supported. Only JPEG, PNG, and WEBP are allowed."
        )
    
    print(f"Processing {filename} ({content_type})")

    # Load the image data
    contents = await file.read()
    image = bytes_to_image(contents)
    if image is None:
        raise HTTPException(
            status_code = 500,
            detail = "Failed to convert image to usable form."
        )

    # Find all bricks in the image
    detected = detector.detect_bricks(image)
    if not detected:
        raise HTTPException(
            status_code = 404,
            detail = "No brick found in the image."
        )

    # The biggest brick is the one we check
    bb_size = [(brick[2]-brick[0]) * (brick[3]-brick[1]) for brick in detected]
    brick = detected[bb_size.index(max(bb_size))]

    return brick

    # confidence = identifier.identify_brick(image, brick)