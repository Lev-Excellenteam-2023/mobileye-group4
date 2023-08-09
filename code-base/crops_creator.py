from typing import Dict, Any

import cv2
import numpy as np

from consts import CROP_DIR, CROP_RESULT, SEQ, IS_TRUE, IGNOR, CROP_PATH, X0, X1, Y0, Y1, COLOR, SEQ_IMAG, COL, X, Y, \
    GTIM_PATH, RADIUS, IMAG_PATH

from pandas import DataFrame
from PIL import Image


def make_crop(*args, **kwargs):
    """
    The function that creates the crops from the image.
    Your return values from here should be the coordinates of the crops in this format (x0, x1, y0, y1, crop content):
    'x0'  The bigger x value (the right corner)
    'x1'  The smaller x value (the left corner)
    'y0'  The smaller y value (the lower corner)
    'y1'  The bigger y value (the higher corner)
    """
    '''
    for x in args[0]:
        for y in args[1]:
            for color in args[2]:
                for radius in args[3]:
                    pass
    '''

    if args[2] == 'r':
        x0 = args[0] + args[3]*2
        x1 = args[0] - args[3]*2
        y0 = args[1] + args[3]*5
        y1 = args[1] - args[3]*2
        color = args[2]
    else:
        x0 = args[0] + args[3]*2
        x1 = args[0] - args[3]*2
        y0 = args[1] + args[3]*2
        y1 = args[1] - args[3]*5
        color = args[2]

    return int(x0), int(x1), int(y0), int(y1), color


def check_crop(*args, **kwargs):
    """
    Here you check if your crop contains a traffic light or not.
    Try using the ground truth to do that (Hint: easier than you think for the simple cases, and if you found a hard
    one, just ignore it for now :). )
    """

    return True, True


def create_crops(df: DataFrame) -> DataFrame:
    # Your goal in this part is to take the coordinates you have in the df, run on it, create crops from them, save them
    # in the 'data' folder, then check if crop you have found is correct (meaning the TFL is fully contained in the
    # crop) by comparing it to the ground truth and in the end right all the result data you have in the following
    # DataFrame (for doc about each field and its input, look at 'CROP_RESULT')
    #
    # *** IMPORTANT ***
    # All crops should be the same size or smaller!!!

    # creates a folder for you to save the crops in, recommended not must
    if not CROP_DIR.exists():
        CROP_DIR.mkdir()

    # For documentation about each key end what it means, click on 'CROP_RESULT' and see for each value what it means.
    # You wanna stick with this DataFrame structure because its output is the same as the input for the next stages.
    result_df = DataFrame(columns=CROP_RESULT)

    # A dict containing the row you want to insert into the result DataFrame.
    result_template: Dict[Any] = {SEQ: '', IS_TRUE: '', IGNOR: '', CROP_PATH: '', X0: '', X1: '', Y0: '', Y1: '',
                                  COL: ''}
    for index, row in df.iterrows():
        result_template[SEQ] = row[SEQ_IMAG]
        result_template[COL] = row[COLOR]

        # example code:
        # ******* rewrite ONLY FROM HERE *******
        x0, x1, y0, y1, crop = make_crop(row[X], row[Y], row[COLOR], row[RADIUS])
        result_template[X0], result_template[X1], result_template[Y0], result_template[Y1] = x0, x1, y0, y1


        image_path = row[IMAG_PATH]
        original_image = Image.open(image_path)

        cropped_image = original_image.crop((x1, y1, x0, y0))
        resized_image = cropped_image.resize((30, 90))

        crop_path = f'../data/crops/{row[SEQ_IMAG]}_{row[X]}_{row[Y]}_{row[COLOR]}.png'
        result_template[CROP_PATH] = crop_path

        resized_image.save(crop_path)

        result_template[IS_TRUE], result_template[IGNOR] = check_crop(df[GTIM_PATH],
                                                                      crop,
                                                                      'everything_else_you_need_here')
        # ******* TO HERE *******

        # added to current row to the result DataFrame that will serve you as the input to part 2 B).
        result_df = result_df._append(result_template, ignore_index=True)
    return result_df
