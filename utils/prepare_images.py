import os

os.environ["OPENCV_IO_ENABLE_JASPER"] = "true"

import cv2
from pathlib import Path
from src_nowcasting.image_preprocessing import PreProcessImage, PreProcessImageDefish


def main(input_path, output_path):
    
    pre_processor = PreProcessImage()


    for day_path in input_path.glob('*/'):

        print(day_path.stem)
        day_out_path = output_path / day_path.stem 
        if day_out_path.exists():
            continue

        for img_path in day_path.glob('*00.jp2'):

            image_raw = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            image_new = pre_processor.transform_image(image_raw)

            
            if not day_out_path.exists():
                os.makedirs(day_out_path)

            cv2.imwrite(day_out_path / Path(img_path.stem + '.jpg'), image_new)


    print('Done')

if __name__=='__main__':
    main(input_path=Path(r'D:\AllSkyCamIR\Images'), output_path=Path(r'data_out\standard_IR_scaled'))