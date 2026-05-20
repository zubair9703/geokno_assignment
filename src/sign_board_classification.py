# src/signboard_ocr_pipeline.py

from pathlib import Path
import cv2
import numpy as np
import easyocr

from ultralytics.models.sam import SAM3SemanticPredictor

from . import config


class SignboardOCRPipeline:

    def __init__(
        self,
        model_path,
        output_dir=config.CLASSIFICATION_OP,
        conf=0.25,
        half=False,
        gpu=False,
    ):
        output_dir = output_dir.format(config.LOCATIONS['building_lat'],config.LOCATIONS['building_lon'])

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=half,
            save=False,
        )

        self.predictor = SAM3SemanticPredictor(
            overrides=overrides
        )

        # EasyOCR
        self.ocr_reader = easyocr.Reader(
            ['en'],
            gpu=gpu
        )

        # Keywords for classification
        self.commercial_keywords = [
            "hotel",
            "restaurant",
            "cafe",
            "shop",
            "store",
            "mall",
            "medical",
            "clinic",
            "pharmacy",
            "bakery",
            "electronics",
            "fashion",
            "mart",
            "salon",
            "mobile",
            "supermarket",
            "bank",
            "atm",
            "office",
            "hospital",
            "agency",
            "enterprise",
            "traders",
            "solutions",
            "services",
            "hardware",
            "institute"
        ]

    def load_image(self, image_path):

        self.image_path = image_path

        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

        self.predictor.set_image(image_path)

    def segment_signboards(self):

        results = self.predictor(
            text=["a signboard"]
        )

        return results

    def create_combined_mask(self, results):

        h, w = self.image.shape[:2]

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for result in results:

            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            for mask in masks:

                mask = (mask > 0.5).astype(np.uint8)

                combined_mask = np.maximum(
                    combined_mask,
                    mask
                )

        combined_mask = combined_mask * 255

        mask_path = self.output_dir / "combined_mask.png"

        cv2.imwrite(
            str(mask_path),
            combined_mask
        )

        return combined_mask

    def extract_text_from_mask_regions(
        self,
        mask,
        min_area=300,
    ):
        """
        Run OCR separately on each connected mask region
        """

        binary_mask = (mask > 127).astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        all_texts = []

        for idx, contour in enumerate(contours):

            area = cv2.contourArea(contour)

            if area < min_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            crop_img = self.image[y:y+h, x:x+w]

            crop_mask = binary_mask[y:y+h, x:x+w]

            # Apply mask
            masked_crop = cv2.bitwise_and(
                crop_img,
                crop_img,
                mask=crop_mask
            )

            # Preprocess
            gray = cv2.cvtColor(
                masked_crop,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.GaussianBlur(
                gray,
                (3, 3),
                0
            )

            # OCR
            results = self.ocr_reader.readtext(
                gray,
                detail=1,
                paragraph=False
            )

            region_texts = []

            for detection in results:

                bbox, text, confidence = detection

                if confidence < 0.2:
                    continue

                region_texts.append(text)

                print(
                    f"[Region {idx}] "
                    f"Text: {text} | "
                    f"Confidence: {confidence:.3f}"
                )

            if len(region_texts) > 0:

                combined_text = " ".join(region_texts)

                all_texts.append(combined_text)
                

        final_text = "\n".join(all_texts)

        return final_text

    def classify_building(self, text):

        text_lower = text.lower()

        print("Detected Text:", text_lower)

        for keyword in self.commercial_keywords:

            if keyword in text_lower:
                return "commercial"

        return "residential"

    def save_visualization(
        self,
        mask,
        classification,
    ):

        overlay = self.image.copy()

        color_mask = np.zeros_like(self.image)

        color_mask[mask > 0] = (0, 255, 0)

        overlay = cv2.addWeighted(
            overlay,
            1.0,
            color_mask,
            0.5,
            0,
        )

        cv2.putText(
            overlay,
            f"Prediction: {classification}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        p_name = f"classified_output_{config.LOCATIONS['building_lat']}_{config.LOCATIONS['building_lon']}.jpg"

        output_path = (
            self.output_dir /
            p_name
        )

        cv2.imwrite(
            str(output_path),
            overlay
        )

        print(f"Saved: {output_path}")

    def run(self, image_path):

        print("Loading image...")
        self.load_image(image_path)

        print("Running SAM3 segmentation...")
        results = self.segment_signboards()

        print("Creating combined mask...")
        mask = self.create_combined_mask(results)

        print("Running EasyOCR on mask regions...")
        text = self.extract_text_from_mask_regions(mask)

        print("\nDetected Text:")
        print(text)

        classification = self.classify_building(text)

        print(f"\nBuilding Type: {classification}")

        self.save_visualization(
            mask,
            classification
        )

        return {
            "text": text,
            "classification": classification,
        }


if __name__ == "__main__":

    MODEL_PATH = r"D:\geokno_assignment\models\sam3.pt"

    IMAGE_PATH = r"D:\geokno_assignment\src\streetview_images\lat_17.4295349427362_lng_78.33249132516058_p22_fov75.jpg"

    pipeline = SignboardOCRPipeline(
        model_path=MODEL_PATH,
        output_dir="outputs",
        gpu=False,
    )

    result = pipeline.run(IMAGE_PATH)

    print(result)