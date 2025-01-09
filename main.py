import pymupdf
import fitz
import numpy as np
import cv2
from pathlib import Path


def convert_page_to_cv2_image(page: pymupdf.Page) -> np.ndarray:
    pix_map = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # type: ignore
    cv2_img = np.frombuffer(pix_map.samples, dtype=np.uint8).reshape(
        pix_map.h, pix_map.w, pix_map.n
    )
    cv2_img = np.ascontiguousarray(cv2_img[..., [2, 1, 0]])
    return cv2_img


if __name__ == "__main__":
    pdf_paths = list(Path(f"./pdf/").glob("*.pdf"))

    for pdf in pdf_paths:
        pdf_name = pdf

        image_folder = Path(f"./images/{pdf_name.stem}")
        image_folder.mkdir(parents=True, exist_ok=True)

        doc = pymupdf.open(pdf_name)

        for idx, page in enumerate(doc):
            cv2_image = convert_page_to_cv2_image(page)
            cv2.imwrite(f"./images/{pdf_name.stem}/{idx}.png", cv2_image)
