import argparse
import os
import tqdm
import numpy as np
import pandas as pd


def findAllSeqs(dirName, extension, speaker_level):
    """
    :param dirName:
    :param extension:
    :param speaker_level:
    :return:

    outSequences:
    Lists all the sequences with the given extension in the dirName directory
    the example of outSequences: outSequences[0]=(0, '/public/home/xiaok/xiaok/data/voxceleb2/id03145/5DdGz9cnGw8/00010.wav')

    outSpeakers:
    The speaker labels at the speaker_level
    the example of outSpeakers: outSpeakers[0] = id03145
    """
    if dirName[-1] != os.sep:
        dirName += os.sep
    prefixSize = len(dirName)
    speakersTarget = {}
    outSequences = []
    for root, dirs, filenames in tqdm.tqdm(os.walk(dirName, followlinks=True)):
        filtered_files = [f for f in filenames if f.endswith(extension)]
        if len(filtered_files) > 0:
            speakerStr = os.sep.join(root[prefixSize:].split(os.sep)[:speaker_level])
            if speakerStr not in speakersTarget:
                speakersTarget[speakerStr] = len(speakersTarget)
            speaker = speakersTarget[speakerStr]
            for filename in filtered_files:
                full_path = os.path.join(root, filename)
                outSequences.append((speaker, full_path))
    outSpeakers = [None for x in speakersTarget]

    for key, index in speakersTarget.items():
        outSpeakers[index] = key

    print('find {} speakers'.format(len(outSpeakers)))
    print('find {} utterance'.format(len(outSequences)))

    return outSequences, outSpeakers


if __name__ == '__main__':
    """
    make the train.csv at data_csv_path
    the example of train.csv:
    id03145 /public/home/xiaok/xiaok/data/voxceleb2/id03145/5DdGz9cnGw8/00010.wav 0
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--extension', help='file extension name', type=str, default='wav')
    parser.add_argument('--dataset_dir', help='dataset dir', type=str, default='')
    parser.add_argument('--data_csv_path', help='csv save path', type=str, default='')
    parser.add_argument('--data_list_path', help='list save path', type=str, default='')
    parser.add_argument('--speaker_level', help='', type=int, default=1)
    args = parser.parse_args()
    outSequences, outSpeakers = findAllSeqs(dirName=args.dataset_dir, extension=args.extension, speaker_level=1)
    outSequences = np.array(outSequences, dtype=str)
    utt_spk_int_labels = outSequences.T[0].astype(int)
    utt_pahts = outSequences.T[1]
    utt_spk_str_labels = []
    for i in utt_spk_int_labels:
        utt_spk_str_labels.append(outSpeakers[i])

    csv_dict = {'speaker_name': utt_spk_str_labels, 'utt_paths': utt_pahts, 'utt_spk_int_labels': utt_spk_int_labels}
    df = pd.DataFrame(data=csv_dict)

    try:
        f = open(args.data_list_path, 'w')
        n = len(utt_spk_int_labels)
        for i in range(n):
            f.write('{} {} {}\n'.format(utt_spk_str_labels[i], utt_pahts[i], utt_spk_int_labels[i]))

        df.to_csv(args.data_csv_path)
        print('Save data csv file at {}'.format(args.data_csv_path))
    except OSError as err:
        print('Ran in an error while saving {}: {}'.format(args.data_csv_path, err))
