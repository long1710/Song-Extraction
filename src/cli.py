import argparse
import json

with open('config_file.json', 'r') as myfile:
    data=myfile.read()

obj = json.loads(data)
def get_args():
    '''
    Required:
    p: path to 5 second video file

    Optional:
    -m: model to select, 10 model available, default to DenseNet201
    -v: version of the code
    :return:
    '''

    parser = argparse.ArgumentParser(
        prog='song_extraction',
        description='Song and Speech classification in 5 seconds audio fragment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '-m',
        metavar='model',
        type=str,
        default="VGG19",
        help= f'available model: {obj["model"].keys()}'
    )

    parser.add_argument('--version', action='version', version= f'%(prog)s {obj["version"]}')

    parser.add_argument('p',
                        metavar='song_path',
                        type=str,
                        help='path to 5 seconds audio classification')

    return parser.parse_args()
