from logging import Logger, INFO, DEBUG, ERROR
import pandas as pd

from typing import Optional


class FileIO(object):
    """Abstract class defining the file I/O handler methods"""

    def __init__(self, logger: Optional[Logger] = None):
        """
        Constructor method to create a FileIO handler object

        Parameters
        ----------
        logger : `Logger` object, optional
            logging handler object to instantiate FileIO object
            with the ability to log progress updates
        """
        self.logger = logger

    def set_logger(self, logger: Optional[Logger] = None):
        """
        Setter method to assign a logging handler to the FileIO object

        Parameters
        ----------
        logger : `Logger` object, optional
            logging handler object to be used with the FileIO object
            to log progress updates
        """
        self.logger = logger

    def log(self, string, mode=INFO):
        """
        Write specified string with preset logging object
        using the mode specified

        Parameters
        ----------
        string : str
            string text to be logged
        mode : int, optional
            One of the supported logging message types.
            Currently supported values are logging.INFO, DEBUG, ERROR
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

        Parameters
        ----------
        dir_path : str
            path for directory to be create
        clear_dir : bool, optional
            clear contents on existing directory

        Returns
        -------
        str
            path to the directory created
        """
        raise NotImplementedError

    def read_df(
        self, infile: str, sep: str = ",", index_col: int = None
    ) -> Optional[pd.DataFrame]:
        """
        Load a pandas dataframe from a file

        Parameters
        ----------
        infile : str
            path to the csv input file
        sep : str, optional
            separator to use for loading file
        index_col : int, optional
            column to be used as index

        Returns
        -------
        `pandas.DataFrame`
            pandas dataframe loaded from specified path
        """
        raise NotImplementedError

    def read_df_list(self, infiles, sep=",", index_col=None) -> pd.DataFrame:
        """
        Load a pandas dataframe from a list of files by concatenating
        the individual dataframes from each file

        Parameters
        ----------
        infiles : list of str
            list of paths to the csv input files
        sep : str, optional
            separator to use for loading file
        index_col : int, optional
            column to be used as index

        Returns:
        `pandas.DataFrame`
            pandas dataframe loaded from specified path
        """
        raise NotImplementedError

    def write_df(self, df, outfile: str = None, sep: str = ",", index: bool = True):
        """
        Write a pandas dataframe to a file

        Parameters
        ----------
        df : `pandas.DataFrame`
            dataframe to be written
        outfile : str. optional
            path to the csv output file
        sep : str, optional
            separator to use for loading file
        index : bool, optional
            boolean specifying if index should be saved
        """
        raise NotImplementedError

    def read_text_file(self, infile) -> str:
        """
        Read text file and return as string

        Parameters
        ----------
        infile : str
            path to the text file

        Returns
        -------
        str
            file contents as a string
        """
        raise NotImplementedError

    def read_json(self, infile) -> dict:
        """
        Read JSON file and return a python dictionary

        Parameters
        ----------
        infile : str
            path to the json file

        Returns
        -------
        dict
            python dictionary loaded from JSON file
        """
        raise NotImplementedError

    def read_yaml(self, infile) -> dict:
        """
        Read YAML file and return a python dictionary

        Parameters
        ----------
        infile : str
            path to the YAML file

        Returns
        -------
        dict
            python dictionary loaded from JSON file
        """
        raise NotImplementedError

    def write_json(self, json_dict: dict, outfile: str):
        """
        Write dictionary to a JSON file

        Parameters
        ----------
        json_dict : dict
            dictionary to be dumped to json file
        outfile : str
            path to the output file
        """
        raise NotImplementedError

    def path_exists(self, path: str) -> bool:
        """
        Check if a file path exists

        Parameters
        ----------
        path : str
            check if path exists

        Returns
        -------
        bool
            True if path exists; False otherwise
        """
        raise NotImplementedError

    def get_files_in_directory(self, indir: str, extension=".csv", prefix=""):
        """
        Get list of files in a directory

        Parameters
        ----------
        indir : str
            input directory to search for files
        extension : str, optional
            extension of the files to search for
        prefix : str, optional
            string file name prefix to narrow search

        Returns
        -------
        list of str
            list of file path strings
        """
        raise NotImplementedError

    def clear_dir(self, dir_path: str):
        """
        Clear contents of existing directory

        Parameters
        ----------
        dir_path : str
            path to directory to be cleared
        """
        raise NotImplementedError

    def rm_dir(self, dir_path: str):
        """
        Delete existing directory

        Parameters
        ----------
        dir_path : str
            path to directory to be removed
        """
        raise NotImplementedError

    def rm_file(self, file_path: str):
        """
        Deletes existing file_path

        Parameters
        ----------
        file_path : str
            path to file to be removed
        """
        raise NotImplementedError
