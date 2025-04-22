import cv2 # OpenCV library
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

IMG_ROWS, IMG_COLS = 28, 28 
MODEL_INPUT_SHAPE = (1, IMG_ROWS, IMG_COLS, 1) 
TARGET_DIGIT_PIXELS = 20 


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)


def preprocess_image(image_bytes: bytes, max_size_bytes: int) -> np.ndarray | None:
    logger.info(f"Starting image preprocessing for image of size {len(image_bytes)} bytes.")

    if len(image_bytes) == 0:
        logger.warning("Preprocessing failed: Input image bytes are empty.")
        return None
    if len(image_bytes) > max_size_bytes:
        logger.warning(f"Preprocessing failed: Image size ({len(image_bytes)} bytes) exceeds limit ({max_size_bytes} bytes).")
        return None

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        if img_bgr is None:
            logger.warning("Preprocessing failed: cv2.imdecode returned None. Invalid image format or corrupted data.")
            return None

        logger.info(f"Image decoded successfully. Shape: {img_bgr.shape}, dtype: {img_bgr.dtype}")

        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 4:
            logger.info("Image has alpha channel, converting to BGR.")
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
        elif len(img_bgr.shape) == 2: #Already grayscale
             logger.info("Image is already grayscale, converting to BGR for consistency.")
             img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif len(img_bgr.shape) != 3 or img_bgr.shape[2] != 3:
             logger.warning(f"Preprocessing failed: Unexpected image shape {img_bgr.shape}")
             return None


    except Exception as e:
        logger.error(f"Preprocessing failed during image decoding: {e}", exc_info=True)
        return None

    try:
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        logger.info(f"Converted to grayscale. Shape: {img_gray.shape}")
    except Exception as e:
        logger.error(f"Preprocessing failed during grayscale conversion: {e}", exc_info=True)
        return None

    try:
        img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
        logger.info("Applied Gaussian blur.")
    except Exception as e:
        logger.error(f"Preprocessing failed during Gaussian blur: {e}", exc_info=True)
        return None

    #Thresholding (Otsu + Binary Step) 
    try:
        ret, thresh_otsu = cv2.threshold(
            img_blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        logger.info(f"Applied Otsu's thresholding. Optimal threshold determined: {ret}")

        ret_final, thresh = cv2.threshold(thresh_otsu, 127, 255, cv2.THRESH_BINARY)
        logger.info("Applied extra binary threshold step for cleanup.")

    except Exception as e:
        logger.error(f"Preprocessing failed during thresholding: {e}", exc_info=True)
        return None

    try:
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.info(f"Found {len(contours)} contours.")

        if not contours:
            logger.warning("Preprocessing failed: No contours found after thresholding.")
            return None

        best_contour = max(contours, key=cv2.contourArea)

        x, y, w, h = cv2.boundingRect(best_contour)
        logger.info(f"Bounding box of largest contour: x={x}, y={y}, w={w}, h={h}")

        if w <= 0 or h <= 0:
             logger.warning("Preprocessing failed: Invalid bounding box dimensions.")
             return None

    except Exception as e:
        logger.error(f"Preprocessing failed during contour detection: {e}", exc_info=True)
        return None

    roi = thresh[y:y+h, x:x+w]
    logger.info(f"Cropped ROI shape: {roi.shape}")

    try:
        current_h, current_w = roi.shape
        aspect_ratio = current_w / current_h

        if current_w > current_h:
            new_w = TARGET_DIGIT_PIXELS
            new_h = int(new_w / aspect_ratio)
        else:
            new_h = TARGET_DIGIT_PIXELS
            new_w = int(new_h * aspect_ratio)

        new_w = max(1, new_w)
        new_h = max(1, new_h)

        resized_roi = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized ROI to: ({new_w}, {new_h})")

        canvas = np.zeros((IMG_ROWS, IMG_COLS), dtype=np.uint8)

        pad_y = (IMG_ROWS - new_h) // 2
        pad_x = (IMG_COLS - new_w) // 2

        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized_roi
        logger.info("Placed resized ROI onto 28x28 canvas.")

    except Exception as e:
        logger.error(f"Preprocessing failed during resize/padding: {e}", exc_info=True)
        return None

    try:
        _, canvas_thresh = cv2.threshold(canvas, 1, 255, cv2.THRESH_BINARY)
        logger.info("Applied final threshold to canvas to ensure binary values.")
        canvas = canvas_thresh 
    except Exception as e:
        logger.error(f"Preprocessing failed during final canvas thresholding: {e}", exc_info=True)
        return None

    processed_image = canvas.astype('float32') / 255.0

    processed_image = np.reshape(processed_image, MODEL_INPUT_SHAPE)
    logger.info(f"Final processed image shape: {processed_image.shape}")

    logger.info("Image preprocessing completed successfully.")
    return processed_image

#Direct testing for preprocessing
""" 
if __name__ == "__main__":
    test_image_path = os.path.join(project_root, 'my_digit.png') 
    max_size = 5 * 1024 * 1024 

    if os.path.exists(test_image_path):
        logger.info(f"--- Testing preprocessing with image: {test_image_path} ---")
        with open(test_image_path, 'rb') as f:
            test_image_bytes = f.read()

        final_image_array = preprocess_image(test_image_bytes, max_size)

        if final_image_array is not None:
            logger.info("--- Preprocessing test completed successfully ---")
            logger.info(f"Output array shape: {final_image_array.shape}") # Should be (1, 28, 28, 1)
            logger.info(f"Output array dtype: {final_image_array.dtype}") # Should be float32
            logger.info(f"Output array min value: {np.min(final_image_array)}") # Should be >= 0.0
            logger.info(f"Output array max value: {np.max(final_image_array)}") # Should be <= 1.0


            vis_image = np.reshape(final_image_array, (IMG_ROWS, IMG_COLS)) * 255.0
            cv2.imwrite("test_processed_output_final_thresh.png", vis_image.astype(np.uint8)) # Changed filename again
            logger.info("Saved processed image for visualization to test_processed_output_final_thresh.png")
        else:
            logger.info("--- Preprocessing test returned None ---")
    else:
        logger.warning(f"Test image not found at {test_image_path}. Provide a valid path to test.")
"""
