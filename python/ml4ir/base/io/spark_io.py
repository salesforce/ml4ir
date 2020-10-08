import json
import yaml
import pandas as pd
from pyspark.sql import SparkSession
from logging import Logger

from ml4ir.base.io.file_io import FileIO

from typing import Optional


class SparkIO(FileIO):
    """Class defining the file I/O handler methods for the HDFS file system using spark"""

    def __init__(self, logger: Optional[Logger] = None):
        """
        Constructor method to create a FileIO handler object and
        set up spark session and hadoop file system handlers

        Parameters
        ----------
        logger : `Logger` object, optional
            logging handler object to instantiate FileIO object
            with the ability to log progress updates
        """
        self.logger = logger
        self.spark_session = SparkSession.builder.appName("ml4ir").getOrCreate()
        self.spark_context = self.spark_session.sparkContext
        self.hadoop_config = self.spark_context._jsc.hadoopConfiguration()
        self.hdfs = self.spark_context._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(
            self.hadoop_config
        )
        self.local_fs = self.hdfs.getLocal(self.hadoop_config)

    def get_path_from_str(self, file_path: str):
        """
        Get Path object from string

        Parameters
        ----------
        file_path : str
            string file path

        Returns
        -------
        hadoop path
            Hadoop Path object
        """
        return self.spark_context._gateway.jvm.org.apache.hadoop.fs.Path(file_path)

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
        self.log("Reading dataframe from : {}".format(infile))
        return (
            self.spark_session.read.format("csv")
            .option("header", "true")
            .option("inferschema", "true")
            .option("mode", "DROPMALFORMED")
            .option("mergeSchema", "true")
            .load(infile)
            .toPandas()
        )

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
        `pandas.DataFrame`
            pandas dataframe loaded from list of files

        Notes
        -----
        `sep` and `index_col` are not used in SparkIO
        """
        self.log("Reading {} files from [{}, ..".format(len(infiles), infiles[0]))
        return (
            self.spark_session.read.format("csv")
            .option("header", "true")
            .option("inferschema", "true")
            .option("mode", "DROPMALFORMED")
            .option("mergeSchema", "true")
            .load(infiles)
            .toPandas()
        )

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
        return "\n".join(
            [row["value"] for row in self.spark_session.read.format("text").load(infile).collect()]
        )

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
        self.log("Reading JSON file : {}".format(infile))
        return json.loads(self.read_text_file(infile))

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
        self.log("Reading YAML file : {}".format(infile))
        return yaml.safe_load(self.read_text_file(infile))

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
        if self.hdfs.exists(self.get_path_from_str(path)):
            self.log("Path exists : {}".format(path))
            return True
        else:
            self.log("Path does not exist : {}".format(path))
            return False

    def rm_dir(self, dir_path: str):
        """
        Delete existing directory

        Parameters
        ----------
        dir_path : str
            path to directory to be removed
        """
        if self.path_exists(dir_path):
            self.hdfs.delete(self.get_path_from_str(dir_path), True)
            self.log("Directory deleted : {}".format(dir_path))

    def rm_file(self, file_path: str):
        """
        Deletes existing file_path

        Parameters
        ----------
        file_path : str
            path to file to be removed
        """
        if self.path_exists(file_path):
            self.hdfs.delete(self.get_path_from_str(file_path), True)
            self.log("File deleted : {}".format(file_path))

    def copy_from_hdfs(self, src: str, dest: str):
        """
        Copy a directory/file from HDFS to local filesystem

        Parameters
        ----------
        src : str
            String path to source(on HDFS)
        dest : str
            String path to destination(on local file system)
        """
        self.hdfs.copyToLocalFile(self.get_path_from_str(src), self.get_path_from_str(dest))

        self.log("Finished copying files from {} to {}".format(src, dest))

    def copy_to_hdfs(self, src: str, dest: str, overwrite=True):
        """
        Copy a directory/file to HDFS from local filesystem

        Parameters
        src : str
            String path to source(on local file system)
        dest : str
            String path to destination(on HDFS)
        overwrite : bool, optional
            Boolean to specify whether existing destination files should be overwritten
        """
        if overwrite:
            self.rm_dir(dest)

        self.hdfs.copyFromLocalFile(self.get_path_from_str(src), self.get_path_from_str(dest))

        self.log("Finished copying files from {} to {}".format(src, dest))
