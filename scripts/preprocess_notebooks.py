"""Simple interface for automating basic tasks with IPYNBs,
using the preprocessors submodule of nbconvert.
Provide first argument from PREPROCESSORS.keys() and then
N>0 arguments that are filenames for notebooks to preprocess.
"""
import argparse
import os.path
import sys

import nbformat
import nbconvert.preprocessors


def main(PREPROCESSOR_STR, NOTEBOOK_PATHS):
    preprocessors = PREPROCESSORS[PREPROCESSOR_STR]
    if PREPROCESSOR_STR == "to_student":
        output_paths = [make_output_path(notebook_path) for notebook_path in NOTEBOOK_PATHS]
    else:
        output_paths = NOTEBOOK_PATHS

    for notebook_path, output_path in zip(NOTEBOOK_PATHS, output_paths):
        do_preprocessing(preprocessors, notebook_path, output_path)


def do_preprocessing(preprocessors, notebook_path, output_path):

    output_directory = os.path.dirname(output_path)

    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    print("Preprocessing {0}".format(notebook_path))

    for preprocessor in preprocessors:
        preprocessor.preprocess(nb, {'metadata': {'path': output_directory}})

    with open(output_path, 'wt') as f:
        nbformat.write(nb, f)

    print("\tSaved to {0}".format(output_path))


def make_execute_preprocessor():
    ep = nbconvert.preprocessors.ExecutePreprocessor(
                                timeout=600, kernel_name='neur299')
    return ep


def make_remove_answers_preprocessor():
    text_answer_pattern = r"""<font color=['"]#?1874CD['"]>"""
    code_answer_pattern = r"""#* answer"""
    rap = nbconvert.preprocessors.RegexRemovePreprocessor(
                                patterns=[code_answer_pattern,
                                          text_answer_pattern,
                                          ])
    return rap


def make_to_student_preprocessors():
    tsps = [make_clear_output_preprocessor(),
            make_remove_answers_preprocessor()
            ]
    return tsps


def make_clear_output_preprocessor():
    cop = nbconvert.preprocessors.ClearOutputPreprocessor()
    return cop


def make_output_path(notebook_path):
    dirname, basename = os.path.split(notebook_path)
    newbasename = ''.join(basename.split(" - Solutions"))
    return os.path.join(dirname, newbasename)


PREPROCESSORS = {"execute": [make_execute_preprocessor()],
                 "remove_answers": [make_remove_answers_preprocessor()],
                 "clear_output": [make_clear_output_preprocessor()],
                 "to_student": make_to_student_preprocessors(),
                               }


if __name__ == "__main__":
    PREPROCESSOR_STR = sys.argv[1]
    assert PREPROCESSOR_STR in PREPROCESSORS.keys()
    NOTEBOOK_PATHS = sys.argv[2:]
    sys.exit(main(PREPROCESSOR_STR, NOTEBOOK_PATHS))
