import os
import json
import shutil
import pandas as pd
import gzip
import sys
import csv
import glob
import yaml

from typing import Optional

from ml4ir.base.io import spark_io

from io import StringIO  # type: ignore

HDFS_PREFIX = "hdfs://"


def make_directory(dir_path: str, clear_dir: bool = False, log=None) -> str:
    """
    Create directory structure specified recursively

    Args:
        dir_path: path for directory to be create
        clear_dir: clear contents on existing directory

    Returns:
        directory path
    """

    if os.path.exists(dir_path):
        if log:
            log.info("Directory already exists : {}".format(dir_path))
        if clear_dir:
            if log:
                log.info("Clearing directory contents")
            try:
                shutil.rmtree(dir_path)
            except shutil.Error as e:  # Potentially OSError, IOError
                raise Exception("Cannot remove local folder, with error: {}".format(e))
        else:
            return dir_path

    # Knowing that folder does not exist, create it from scratch
    os.makedirs(dir_path)

    return dir_path


def read_df(
    infile: str, sep: str = ",", index_col: int = None, log_path: bool = False, log=None
) -> Optional[pd.DataFrame]:
    """
    Load a pandas dataframe from a file

    Args:
        infile: path to the csv input file; can be hdfs path
        sep: separator to use for loading file
        index_col: column to be used as index
        log_path: boolean specifying whether path should be logged
        log: logging object

    Returns:
        pandas dataframe
    """
    if log_path:
        log.info("Loading dataframe from path : {}".format(infile))

    if infile.startswith(HDFS_PREFIX):
        # NOTE: Move to fully spark dataframe based operations in the future
        return spark_io.read_df(infile).toPandas()
    elif infile.endswith(".gz"):
        fp = gzip.open(os.path.expanduser(infile), "rb")
    else:
        fp = open(os.path.expanduser(infile), "r")

    # Redirect stderr to custom string IO
    stderr_old = sys.stderr
    sys.stderr = bad_lines_io = StringIO()

    try:
        df: pd.DataFrame = pd.read_csv(
            fp,
            sep=sep,
            index_col=index_col,
            skipinitialspace=True,
            quotechar='"',
            escapechar="\\",
            error_bad_lines=False,
            warn_bad_lines=True,
            engine="c",
        )
    except Exception as e:
        if log:
            log.info("Error while reading : {}\n{}".format(fp, e))
        return None

    # Get the bad line string value and close the string IO
    bad_lines = bad_lines_io.getvalue()
    bad_lines_io.close()
    sys.stderr = stderr_old

    # Log any bad lines
    if bad_lines and log:
        log.info("Bad lines were found in the file : {}\n{}".format(infile, bad_lines))

    fp.close()
    return df


def read_df_list(infiles, sep=",", index_col=None, log_path=True, log=None) -> pd.DataFrame:
    """
    Load a pandas dataframe from a list of files

    Args:
        infiles: paths to the csv input files; can be hdfs paths
        sep: separator to use for loading file
        index_col: column to be used as index
        log_path: boolean specifying whether path should be logged
        log: logging object

    Returns:
        pandas dataframe
    """
    if log:
        log.info("Reading {} files from [{}, ..".format(len(infiles), infiles[0]))
    return pd.concat(
        [
            read_df(infile, sep=sep, index_col=index_col, log_path=False, log=log)
            for infile in infiles
        ]
    )


def write_df(df, outfile: str = None, sep: str = ",", index: bool = True, log=None) -> str:
    """
    Write a pandas dataframe to a file

    Args:
        df: dataframe to be written
        outfile: path to the csv output file; can NOT be hdfs path currently
        sep: separator to use for loading file
        index: boolean specifying if index should be saved
        log: logging object

    Returns:
        dataframe in csv form if outfile is None

    NOTE: Does not support spark write
    """
    if outfile and outfile.startswith(HDFS_PREFIX):
        raise NotImplementedError
    else:
        output = df.to_csv(
            sep=sep, index=index, quotechar='"', escapechar="\\", quoting=csv.QUOTE_NONNUMERIC
        )
        output = output.replace("\\", "\\\\")

        if outfile:
            fp = open(outfile, "w")
            fp.write(output)
            fp.close()
        return output


def read_json(infile, log=None) -> dict:
    """
    Read JSON file and return a python dictionary

    Args:
        infile: path to the json file; can be hdfs path

    Returns:
        python dictionary
    """
    if log:
        log.info("Reading JSON file from : {}".format(infile))
    if infile.startswith(HDFS_PREFIX):
        return spark_io.read_json(infile)
    else:
        return json.load(open(infile, "r"))


def read_yaml(infile, log=None) -> dict:
    """
    Read YAML file and return a python dictionary

    Args:
        infile: path to the json file; can be hdfs path

    Returns:
        python dictionary
    """
    if log:
        log.info("Reading YAML file from : {}".format(infile))
    if infile.startswith(HDFS_PREFIX):
        return spark_io.read_yaml(infile)
    else:
        return yaml.safe_load(open(infile, "r"))


def write_json(json_dict: dict, outfile: str, log=None):
    """
    Write dictionary to a JSON file

    Args:
        json_dict: dictionary to be dumped to json file
        outfile: path to the output file
        log: logging object
    """
    if outfile.startswith(HDFS_PREFIX):
        # hdfs.dump(json.dumps(json_dict, indent=4, sort_keys=True), outfile)
        raise NotImplementedError
    else:
        json.dump(json_dict, open(outfile, "w"), indent=4, sort_keys=True)


def path_exists(path: str, log=None) -> bool:
    """
    Check if a path exists

    Args:
        path: check if path exists

    Returns:
        True if path exists; False otherwise
    """
    if path.startswith(HDFS_PREFIX):
        # return hdfs.path.exists(path)
        raise NotImplementedError
    else:
        return os.path.exists(path)


def get_files_in_directory(indir: str, extension=".csv", prefix="", log=None):
    """
    Get list of files in a directory

    Args:
        indir: input directory to search for files
        extension: extension of the files to search for
        prefix: string file name prefix to narrow search
        log: logging object

    Returns:
        list of file path strings
    """
    if indir.startswith(HDFS_PREFIX):
        # NOTE: Current implementation just returns the directory wrapped in a list
        return [indir]
    else:
        return sorted(glob.glob(os.path.join(indir, "{}*{}".format(prefix, extension))))


def clear_dir(dir_path: str, log=None):
    """
    Clear contents of existing directory

    Args:
        dir_path: path to directory to be cleared
    """
    if dir_path.startswith(HDFS_PREFIX):
        raise NotImplementedError
    else:
        for dir_content in glob.glob(os.path.join(dir_path, "*")):
            if os.path.isfile(dir_content):
                os.remove(dir_content)
            elif os.path.isdir(dir_content):
                shutil.rmtree(dir_content)


def rm_dir(dir_path: str, log=None):
    """
    Delete existing directory

    Args:
        dir_path: path to directory to be removed
    """
    if dir_path.startswith(HDFS_PREFIX):
        raise NotImplementedError
    else:
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)


def rm_file(file_path: str, log=None):
    """
    Deletes existing file_path

    Args:
        file_path: path to file to be removed
    """
    if file_path.startswith(HDFS_PREFIX):
        raise NotImplementedError
    else:
        if os.path.isfile(file_path):
            os.remove(file_path)
