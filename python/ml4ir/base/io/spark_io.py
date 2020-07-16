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

    def get_path_from_str(self, file_path: str):
        """
        Get Path object from string

        Args:
            file_path: string file path

        Returns:
            Hadoop Path object
        """
        return self.spark_context._gateway.jvm.org.apache.hadoop.fs.Path(file_path)

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
        self.hdfs.copyToLocalFile(self.get_path_from_str(src), self.get_path_from_str(dest))

        self.log("Finished copying files from {} to {}".format(src, dest))

    def copy_to_hdfs(self, src: str, dest: str, overwrite=True):
        """
        Copy a directory/file to HDFS from local filesystem

        Args:
            - src: String path to source(on local file system)
            - dest: String path to destination(on HDFS)
            - overwrite: Boolean to specify whether existing destination files should be overwritten
        """
        if overwrite:
            self.rm_dir(dest)

        self.hdfs.copyFromLocalFile(self.get_path_from_str(src), self.get_path_from_str(dest))

        self.log("Finished copying files from {} to {}".format(src, dest))
