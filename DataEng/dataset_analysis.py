# importing pandas module
import os

import pandas as pd

from DataEng.Preprocessor import Preprocessor


def generate_dataframe(dir: str = os.getcwd(), preprocessor: Preprocessor = None) -> pd.DataFrame:
    """
    generates a dataframe wich shows the number of images for each subdataset, if a preprocessor is given only finds
    the files in which no hands are detected

    :param dir: directory to the dataset
    :param preprocessor:
    :return:
    """
    total_entries_per_class_per_subds = {}
    individual_subds = []

    subdirs = [f.path for f in os.scandir(dir) if f.is_dir()]
    for subdirectory in subdirs:
        sub_dir_name = subdirectory.split('/')[-1]
        subds = {}
        for f in [f.path for f in os.scandir(subdirectory) if not f.is_dir()]:
            subds_name = f.split('/')[-1][0]
            if subds_name not in individual_subds:
                individual_subds.append(subds_name)
            if not preprocessor:
                if subds_name in list(subds.keys()):
                    subds[subds_name] += 1
                else:
                    subds[subds_name] = 1
            else:
                if preprocessor.check_if_hand_is_present(f):
                    if subds_name in list(subds.keys()):
                        subds[subds_name] += 1
                    else:
                        subds[subds_name] = 1
        total_entries_per_class_per_subds[sub_dir_name] = subds

    # now generate the pandas dataframe
    origin = []
    rock = []
    paper = []
    scissors = []
    for subds in individual_subds:
        origin.append(subds)
        rock.append(total_entries_per_class_per_subds['rock'][subds])
        paper.append(total_entries_per_class_per_subds['paper'][subds])
        scissors.append(total_entries_per_class_per_subds['scissors'][subds])

    data = {'Origin': origin,
            'Rock': rock,
            'Paper': paper,
            'Scissors': scissors}

    return pd.DataFrame(data)


def analyze_findig_hands(dir: str, hands_detection_confidence=[.1, .2, .3, .4, .5]):
    ret = {}
    for confidence in hands_detection_confidence:
        preprocessor = Preprocessor(hands_detection_confidence=confidence)
        ret[str(confidence)] = generate_dataframe(dir, preprocessor)
    return ret


def expand_df(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df.loc['Class_Total'] = df.sum(numeric_only=True, axis=0)
    df.loc[:, 'Dataset_Total'] = df.sum(numeric_only=True, axis=1)
    return df


def calculate_percentage(frame_instance: pd.DataFrame, frame_total: pd.DataFrame):
    f_i = frame_instance.iloc[:, 1:]
    f_t = frame_total.iloc[:, 1:]
    return pd.DataFrame(f_i.values / f_t.values, columns=f_i.columns, index=f_i.index)


def analyse_outcome():
    analysis_folder = '/Users/amling/uni/shifumi/DataEng/analysis'

    total = pd.read_csv(os.path.join(analysis_folder, "total.csv"))
    print(expand_df(total))
    confidence_df = {}

    for i in range(5):
        hand_confidence = (i + 1) / 10
        frame = pd.read_csv(os.path.join(analysis_folder, f'dot{i + 1}_detection.csv'))
        confidence_df[hand_confidence] = frame

    for hand_confidence, frame in zip(confidence_df, confidence_df.values()):
        print(
            f'hand_confidence: {hand_confidence} \n{expand_df(frame)}\n{calculate_percentage(expand_df(frame), expand_df(total))}')


# ds = '/Users/amling/uni/shifumi/DataEng/datasets/combined/combined'
# for this dataset the outcomnes are already stored in analysis
# read in total from csv

analyse_outcome()
