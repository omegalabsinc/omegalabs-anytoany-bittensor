import time
import replicate
from replicate.exceptions import ModelError, ReplicateError
from concurrent.futures import ThreadPoolExecutor

import time
import replicate
from replicate.exceptions import ModelError, ReplicateError

def embed_modality(f, modality="video", timeout=60):
    start = time.time()
    print(f"input is {f}")

    try:
        # Get the latest version of the model
        model = replicate.models.get("omegalabsinc/imagebind")
        version = model.versions.list()[0]

        # Start the prediction
        prediction = replicate.predictions.create(
            version=version,
            input={
                "input": f,
                "modality": modality
            }
        )

        # Wait for the prediction to complete or timeout
        while prediction.status not in ["succeeded", "failed", "canceled"]:
            time.sleep(1)
            prediction.reload()
            if time.time() - start > timeout:
                print(f"Operation timed out after {timeout} seconds. Canceling prediction.")
                replicate.predictions.cancel(prediction.id)
                return None

        if prediction.status == "succeeded":
            output = prediction.output
            print(f"Got output of length {len(output)} in {time.time() - start} seconds.")
            return output
        else:
            print(f"Prediction failed with status: {prediction.status}")
            return None

    except (ModelError, ReplicateError) as e:
        print(f"Error: {e}")
        if hasattr(e, 'prediction') and e.prediction:
            print(f"Error message: {e.prediction.detail}")
            print(f"Failed prediction: {e.prediction.id}")
        return None

def embed_image(f):
    start = time.time()
    print(f"input is {f}")
    output = replicate.run(
        "omegalabsinc/imagebind:4e65d8812e4327042e264cac30f3c251c53bd0d781625c78c9a7348bd666353f",
        input={
            "input": f,
            "modality": "vision"
        }
    )
    print(f"Got output of length {len(output)} in {time.time() - start} seconds.")

def main():
    #image_file = open("omega_favico.png", "rb")
    #with ThreadPoolExecutor() as executor:
    #    executor.submit(embed_image, image_file)
    
    #time.sleep(3)

    #video_file = open("test.mp4", "rb")
    #video_file = "https://videos.pexels.com/video-files/3756003/3756003-sd_506_960_25fps.mp4"
    video_file = "https://storage.googleapis.com/omega-a2a-mm-chat/3756003-uhd_2160_4096_25fps.mp4"
    embed_modality(video_file, "video")

if __name__ == "__main__":
    main()