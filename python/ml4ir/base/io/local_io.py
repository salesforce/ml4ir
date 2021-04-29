import os
import json
import shutil
import pandas as pd
import numpy as np
import gzip
import sys
import csv
import glob
import yaml
import logging

from ml4ir.base.io.file_io import FileIO

from typing import Optional

from io import StringIO  # type: ignore


class LocalIO(FileIO):
    """Class defining the file I/O handler methods for the local file system"""

    def make_directory(self, dir_path: str, clear_dir: bool = False):
        """
        Create directory structure specified recursively

        Parameters
        ----------
        dir_path : str
            path for directory to be create
        clear_dir : bool, optional
            clear contents on existing directory
        """

        if os.path.exists(dir_path):
            if clear_dir:
                try:
                    shutil.rmtree(dir_path)
                except shutil.Error as e:  # Potentially OSError, IOError
                    raise Exception("Cannot remove local folder, with error: {}".format(e))
            else:
                return dir_path

        # Knowing that folder does not exist, create it from scratch
        os.makedirs(dir_path)

    def read_df(
        self, infile: str, sep: str = ",", index_col: int = None
    ) -> Optional[pd.DataFrame]:
        """
        Load a pandas dataframe from a file

        Parameters
        ----------
        infile : str
            path to the csv input file; can be hdfs path
        sep : str, optional
            separator to use for loading file
        index_col : int, optional
            column to be used as index

        Returns
        -------
        `pandas.DataFrame`
            pandas dataframe loaded from file
        """
        self.log("Loading dataframe from path : {}".format(infile))

        if infile.endswith(".gz"):
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
            self.log("Error while reading : {}\n{}".format(fp, e), mode=logging.WARN)
            return None

        # Get the bad line string value and close the string IO
        bad_lines = bad_lines_io.getvalue()
        bad_lines_io.close()
        sys.stderr = stderr_old

        # Log any bad lines
        if bad_lines:
            self.log("Bad lines were found in the file : {}\n{}".format(infile, bad_lines))

        fp.close()
        return df

    def read_df_list(self, infiles, sep=",", index_col=None) -> pd.DataFrame:
        """
        Load a pandas dataframe from a list of files

        Parameters
        ----------
        infiles : list of str
            paths to the csv input files; can be hdfs paths
        sep : str, optional
            separator to use for loading file
        index_col : int, optional
            column to be used as index

        Returns
        -------
        `pd.DataFrame`
            pandas dataframe loaded from file
        """
        self.log("Reading {} files from [{}, ..".format(len(infiles), infiles[0]))
        return pd.concat(
            [self.read_df(infile, sep=sep, index_col=index_col) for infile in infiles]
        )

    def write_df(self, df, outfile: str = None, sep: str = ",", index: bool = True) -> str:
        """
        Write a pandas dataframe to a file

        Parameters
        ----------
        df : `pandas.DataFrame`
            dataframe to be written
        outfile : str
            path to the csv output file; can NOT be hdfs path currently
        sep : str
            separator to use for loading file
        index : int
            boolean specifying if index should be saved

        Returns
        -------
        str
            dataframe in csv form if outfile is None
        """
        self.log("Writing dataframe to : {}".format(outfile))
        output = df.to_csv(
            sep=sep, index=index, quotechar='"', escapechar="\\", quoting=csv.QUOTE_NONNUMERIC
        )
        output = output.replace("\\", "\\\\")

        if outfile:
            fp = open(outfile, "w")
            fp.write(output)
            fp.close()
            return ""
        else:
            return output

    def read_json(self, infile) -> dict:
        """
        Read JSON file and return a python dictionary

        Parameters
        ----------
        infile : str
            path to the json file; can be hdfs path

        Returns
        -------
        dict
            python dictionary loaded from file
        """
        self.log("Reading JSON file from : {}".format(infile))
        return json.load(open(infile, "r"))

    def read_yaml(self, infile) -> dict:
        """
        Read YAML file and return a python dictionary

        Parameters
        ----------
        infile : str
            path to the json file; can be hdfs path

        Returns
        -------
        dict
            python dictionary loaded from file
        """
        self.log("Reading YAML file from : {}".format(infile))
        return yaml.safe_load(open(infile, "r"))

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
        self.log("Writing JSON dictionary to : {}".format(outfile))
        json.dump(json_dict, open(outfile, "w"), indent=4, sort_keys=True)

    def path_exists(self, path: str) -> bool:
        """
        Check if a path exists

        Parameters
        ----------
        path : str
            check if path exists

        Returns
        -------
        bool
            True if path exists; False otherwise
        """
        if os.path.exists(path):
            self.log("Path exists: {}".format(path))
            return True
        else:
            self.log("Path does not exists: {}".format(path))
            return False

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
        files_in_directory = sorted(
            glob.glob(os.path.join(indir, "{}*{}".format(prefix, extension)))
        )
        self.log("{} files found under {}".format(len(files_in_directory), indir))
        return files_in_directory

    def clear_dir_contents(self, dir_path: str):
        """
        Clear contents of existing directory

        Parameters
        ----------
        dir_path :  str
            path to directory to be cleared
        """
        for dir_content in glob.glob(os.path.join(dir_path, "*")):
            if os.path.isfile(dir_content):
                os.remove(dir_content)
            elif os.path.isdir(dir_content):
                shutil.rmtree(dir_content)
        self.log("Directory cleared : {}".format(dir_path))

    def rm_dir(self, dir_path: str):
        """
        Delete existing directory

        Parameters
        ----------
        dir_path : str
            path to directory to be removed
        """
        if os.path.isdir(dir_path):
            shutil.rmtree(dir_path)
            self.log("Directory deleted : {}".format(dir_path))

    def rm_file(self, file_path: str):
        """
        Delete existing file_path

        Parameters
        ----------
        file_path : str
            path to file to be removed
        """
        if os.path.isfile(file_path):
            os.remove(file_path)
            self.log("File deleted : {}".format(file_path))

    def save_numpy_array(self, np_array, file_path: str, allow_pickle=True, zip=True, **kwargs):
        """
        Save a numpy array to disk

        Parameters
        ----------
        np_array : numpy array or list of numpy arrays
            Array like numpy object to be saved
        file_path : str
            file path to save the object to
        allow_pickle : bool, optional
            Allow pickling of objects while saving
        zip : bool, optional,
            use np.savez to save the numpy arrays, allows passing in python list

        Notes
        -----
        Used to save individual model layer weights for transfer learning.

        If using zip=True, the np_array has to be a python list
        tensorflow layer weights are lists of arrays.
        np.save() can not be used for saving list of numpy arrays directly
        as it tries to manually convert the list into a numpy array, leading
        to errors with numpy shape.
        savez allows us to save each list item in separate files and abstracts this step for end user.
        """
        if zip:
            """
            NOTE: In this case, the np_array has to be a python list

            tensorflow layer weights are lists of arrays.
            np.save() can not be used for saving list of numpy arrays directly
            as it tries to manually convert the list into a numpy array, leading
            to errors with numpy shape.
            savez allows us to save each list item in separate files and abstracts this step for end user.
            """
            np.savez(file_path, *np_array)
        else:
            np.save(file_path, arr=np_array, allow_pickle=allow_pickle, **kwargs)

    def load_numpy_array(self, file_path, allow_pickle=True, unzip=True, **kwargs):
        """
        Load a numpy array from disk

        Parameters
        ----------
        file_path : str
            file path to load the numpy object from
        allow_pickle : bool, optional
            Allow pickling of objects while loading
        unzip : bool, optional
            To unzip the numpy array saved as a zip file. Used when saved with zip=True

        Returns
        -------
        list of numpy arrays
            python list of numpy arrays

        Notes
        -----
        Used to load individual model layer weights for transfer learning
        """
        np_array = np.load(file_path, allow_pickle=allow_pickle, **kwargs)

        if unzip:
            np_array_list = list()
            for np_file in np_array.files:
                np_array_list.append(np_array[np_file])
            return np_array_list
        else:
            return np_array
