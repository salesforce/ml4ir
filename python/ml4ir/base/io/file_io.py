from logging import Logger, INFO, DEBUG, ERROR
import pandas as pd

from typing import Optional


class FileIO(object):
    """Abstract class defining the file I/O handler methods"""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger

    def set_logger(self, logger: Optional[Logger] = None):
        self.logger = logger

    def log(self, string, mode=INFO):
        """
        Wrapper method for logging

        Args:
            string: String to be logged
            mode: logging mode to use
        """
        if self.logger:
            if mode == INFO:
                self.logger.info(string)
            elif mode == DEBUG:
                self.logger.info(string)
            elif mode == ERROR:
                self.logger.info(string)

    def make_directory(self, dir_path: str, clear_dir: bool = False) -> str:
        """
        Create directory structure specified recursively

        Args:
            dir_path: path for directory to be create
            clear_dir: clear contents on existing directory

        Returns:
            directory path
        """
        raise NotImplementedError

    def read_df(
        self, infile: str, sep: str = ",", index_col: int = None
    ) -> Optional[pd.DataFrame]:
        """
        Load a pandas dataframe from a file

        Args:
            infile: path to the csv input file
            sep: separator to use for loading file
            index_col: column to be used as index

        Returns:
            pandas dataframe
        """
        raise NotImplementedError

    def read_df_list(self, infiles, sep=",", index_col=None) -> pd.DataFrame:
        """
        Load a pandas dataframe from a list of files

        Args:
            infiles: paths to the csv input files
            sep: separator to use for loading file
            index_col: column to be used as index

        Returns:
            pandas dataframe
        """
        raise NotImplementedError

    def write_df(self, df, outfile: str = None, sep: str = ",", index: bool = True):
        """
        Write a pandas dataframe to a file

        Args:
            df: dataframe to be written
            outfile: path to the csv output file
            sep: separator to use for loading file
            index: boolean specifying if index should be saved
        """
        raise NotImplementedError

    def read_text_file(self, infile) -> str:
        """
        Read text file and return as string

        Args:
            infile: path to the text file

        Returns:
            file contents as a string
        """
        raise NotImplementedError

    def read_json(self, infile) -> dict:
        """
        Read JSON file and return a python dictionary

        Args:
            infile: path to the json file

        Returns:
            python dictionary
        """
        raise NotImplementedError

    def read_yaml(self, infile) -> dict:
        """
        Read YAML file and return a python dictionary

        Args:
            infile: path to the json file

        Returns:
            python dictionary
        """
        raise NotImplementedError

    def write_json(self, json_dict: dict, outfile: str):
        """
        Write dictionary to a JSON file

        Args:
            json_dict: dictionary to be dumped to json file
            outfile: path to the output file
        """
        raise NotImplementedError

    def path_exists(self, path: str) -> bool:
        """
        Check if a path exists

        Args:
            path: check if path exists

        Returns:
            True if path exists; False otherwise
        """
        raise NotImplementedError

    def get_files_in_directory(self, indir: str, extension=".csv", prefix=""):
        """
        Get list of files in a directory

        Args:
            indir: input directory to search for files
            extension: extension of the files to search for
            prefix: string file name prefix to narrow search

        Returns:
            list of file path strings
        """
        raise NotImplementedError

    def clear_dir(self, dir_path: str):
        """
        Clear contents of existing directory

        Args:
            dir_path: path to directory to be cleared
        """
        raise NotImplementedError

    def rm_dir(self, dir_path: str):
        """
        Delete existing directory

        Args:
            dir_path: path to directory to be removed
        """
        raise NotImplementedError

    def rm_file(self, file_path: str):
        """
        Deletes existing file_path

        Args:
            file_path: path to file to be removed
        """
        raise NotImplementedError
