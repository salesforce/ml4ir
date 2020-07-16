import json
import yaml
import pandas as pd
from pyspark.sql import SparkSession
from logging import Logger

from ml4ir.base.io.file_io import FileIO

from typing import Optional


class SparkIO(FileIO):
    """Abstract class defining the file I/O handler methods"""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.spark_session = SparkSession.builder.appName("ml4ir").getOrCreate()
        self.spark_context = self.spark_session.sparkContext
        self.hadoop_config = self.spark_context._jsc.hadoopConfiguration()
        self.hdfs = self.spark_context._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(
            self.hadoop_config
        )
        self.local_fs = self.hdfs.getLocal(self.hadoop_config)

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
            infile: path to the csv input file; can be hdfs path
            sep: separator to use for loading file
            index_col: column to be used as index

        Returns:
            pandas dataframe
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

        Args:
            infiles: paths to the csv input files; can be hdfs paths
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
            outfile: path to the csv output file; can NOT be hdfs path currently
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
        return "\n".join(
            [row["value"] for row in self.spark_session.read.format("text").load(infile).collect()]
        )

    def read_json(self, infile) -> dict:
        """
        Read JSON file and return a python dictionary

        Args:
            infile: path to the json file; can be hdfs path

        Returns:
            python dictionary
        """
        self.log("Reading JSON file : {}".format(infile))
        return json.loads(self.read_text_file(infile))

    def read_yaml(self, infile) -> dict:
        """
        Read YAML file and return a python dictionary

        Args:
            infile: path to the json file; can be hdfs path

        Returns:
            python dictionary
        """
        self.log("Reading YAML file : {}".format(infile))
        return yaml.safe_load(self.read_text_file(infile))

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
        if self.hdfs.exists(self.get_path_from_str(path)):
            self.log("Path exists : {}".format(path))
            return True
        else:
            self.log("Path does not exist : {}".format(path))
            return False

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
        if self.path_exists(dir_path):
            self.hdfs.delete(self.get_path_from_str(dir_path), True)
            self.log("Directory deleted : {}".format(dir_path))

    def rm_file(self, file_path: str):
        """
        Deletes existing file_path

        Args:
            file_path: path to file to be removed
        """
        if self.path_exists(file_path):
            self.hdfs.delete(self.get_path_from_str(file_path), True)
            self.log("File deleted : {}".format(file_path))

    def copy_from_hdfs(self, src: str, dest: str):
        """
        Copy a directory/file from HDFS to local filesystem

        Args:
            - src: String path to source(on HDFS)
            - dest: String path to destination(on local file system)
        """
        self.log("Copying files from {} to {}".format(src, dest))

        self.hdfs.copyToLocalFile(self.get_path_from_str(src), self.get_path_from_str(dest))

    def copy_to_hdfs(self, src: str, dest: str, overwrite=True):
        """
        Copy a directory/file to HDFS from local filesystem

        Args:
            - src: String path to source(on local file system)
            - dest: String path to destination(on HDFS)
            - overwrite: Boolean to specify whether existing destination files should be overwritten
        """
        self.log("Copying files from {} to {}".format(src, dest))

        if overwrite:
            self.rm_dir(dest)

        self.hdfs.copyFromLocalFile(self.get_path_from_str(src), self.get_path_from_str(dest))
