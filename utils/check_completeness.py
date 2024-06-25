from pathlib import Path
import pandas as pd

def check_completeness(path_in:str, path_out:str, n_images:int) -> None:
    
    #TODO: file overwriting
    
    images_path_ir = Path(path_in)
    output_day = []
    output_pro = []

    # Run through the images
    for i, path in enumerate(images_path_ir.glob('*/')):
        output_day.append(path.stem)
        output_pro.append(len(list(path.glob('*'))) / n_images)

    # Construct and save the output dataframe
    df_output = pd.DataFrame({'date':output_day, 'percent':output_pro})
    df_output.to_csv(path_out)

    return

def main():

    check_completeness(path_in=r'D:\AllSkyCamIR\Images', path_out='count_IR.csv', n_images=2880)
    check_completeness(path_in=r'D:\AllSkyCamVisible\Images', path_out='count_VIS.csv', n_images=1440)

if __name__ == '__main__':
    main()