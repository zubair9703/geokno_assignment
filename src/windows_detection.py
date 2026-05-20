# src/signboard_segmentation_pipeline.py

from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics.models.sam import SAM3SemanticPredictor
from sklearn.cluster import DBSCAN
from . import config

class WindowsSegmentationPipeline:
    def __init__(
        self,
        model_path,
        conf=0.25,
        half=False,
        save=True,
        output_dir=config.WINDOWS_DET_OP,
    ):
        """
        Initialize SAM3 semantic segmentation pipeline.
        """

        output_dir = output_dir.format(config.LOCATIONS['building_lat'],config.LOCATIONS['building_lon'])
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        overrides = dict(
            conf=conf,
            task="segment",
            mode="predict",
            model=model_path,
            half=half,
            save=save,
        )

        self.predictor = SAM3SemanticPredictor(overrides=overrides)

    def load_image(self, image_path):
        """
        Set image for SAM3 predictor.
        """

        self.image_path = image_path
        self.predictor.set_image(image_path)

        self.image = cv2.imread(image_path)

        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")

    def predict(self, concepts):
        """
        Run semantic segmentation using text prompts.

        Example:
            concepts = ["a signboard", "a shop", "a building"]
        """

        if isinstance(concepts, str):
            concepts = [concepts]

        results = self.predictor(text=concepts)

        return results

    def visualize_results(
        self,
        results,
        save_name="prediction.jpg",
    ):
        """
        Visualize YOLO predictions using built-in Ultralytics plotting.
        """

        # Plot predictions
        annotated_image = results[0].plot()

        # Convert BGR -> RGB
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_BGR2RGB,
        )

        # Create figure
        plt.figure(figsize=(12, 12))

        plt.imshow(annotated_image)

        plt.axis("off")

        plt.title("YOLO Inference")

        p_name = f"wd_output_{config.LOCATIONS['building_lat']}_{config.LOCATIONS['building_lon']}.jpg"

        output_path = self.output_dir / save_name

        # Save image
        plt.savefig(
            output_path,
            bbox_inches="tight",
            pad_inches=0,
            dpi=300,
        )

        plt.close()

        print(f"Saved visualization: {output_path}")

        return output_path

    def extract_masks(
        self,
        results,
        save_mask=True,
        mask_name="combined_mask.png",
    ):
        """
        Combine all masks into a single binary mask.
        """

        h, w = self.image.shape[:2]

        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for result in results:

            if result.masks is None:
                continue

            masks = result.masks.data.cpu().numpy()

            for mask in masks:

                mask = (mask > 0.5).astype(np.uint8)

                combined_mask = np.maximum(combined_mask, mask)

        combined_mask = combined_mask * 255

        mask_path = self.output_dir / mask_name

        if save_mask:
            cv2.imwrite(str(mask_path), combined_mask)

        print(f"Saved combined mask: {mask_path}")

        return combined_mask
    
    def estimate_floor_count(self,results, eps=40, min_samples=3):

        centers = []

        for box in results[0].boxes.xyxy.cpu().numpy():

            x1, y1, x2, y2 = box

            cy = (y1 + y2) / 2

            centers.append(cy)

        if len(centers) == 0:
            return 0, []

        centers_array = np.array(centers).reshape(-1, 1)

        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit(centers_array)

        floor_count = len(set(clustering.labels_))

        return floor_count

    def run(
        self,
        image_path,
        concepts,
        save_visualization=True,
        save_masks=True,
    ):
        """
        Complete pipeline.
        """

        print("Loading image...")
        self.load_image(image_path)

        print("Running prediction...")
        results = self.predict(concepts)

        print('Counting no of floors')
        fc = self.estimate_floor_count(results)


        if save_visualization:
            self.visualize_results(results)

        if save_masks:
            self.extract_masks(results)

        print("Pipeline completed.")

        return results,fc


if __name__ == "__main__":

    MODEL_PATH = r"D:\geokno_assignment\models\sam3.pt"

    IMAGE_PATH = r"D:\geokno_assignment\src\streetview_images\lat_17.466868599733957_lng_78.30992252422388_p_22_fov_75.jpg"

    pipeline = WindowsSegmentationPipeline(
        model_path=MODEL_PATH,
        conf=0.5,
        half=False,
        save=True,
        output_dir="outputs",
    )

    results,fc = pipeline.run(
        image_path=IMAGE_PATH,
        concepts=[
            "a signboard",
        ],
        save_visualization=True,
        save_masks=True,
    )