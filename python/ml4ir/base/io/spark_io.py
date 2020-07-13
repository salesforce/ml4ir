import json
import yaml
from pyspark import SparkContext
from pyspark.sql import SparkSession

HDFS_PREFIX = "hdfs://"


class SparkConfigHolder:
    """Class to store configuration objects for Spark to avoid duplication"""

    HADOOP_CONFIG = None
    HDFS = None
    LOCAL_FS = None


def get_spark_session() -> SparkSession:
    """Get or create Spark Session"""
    return SparkSession.builder.appName("ml4ir").getOrCreate()


def get_spark_context() -> SparkContext:
    """Get spark context"""
    return get_spark_session().sparkContext


def get_path_from_str(file_path):
    """Convert a string file path to org.apache.hadoop.fs.Path object"""
    return get_spark_context()._gateway.jvm.org.apache.hadoop.fs.Path(file_path)


def get_hadoop_config():
    """Get hadoop configuration object"""
    if not SparkConfigHolder.HADOOP_CONFIG:
        SparkConfigHolder.HADOOP_CONFIG = get_spark_context()._jsc.hadoopConfiguration()

    return SparkConfigHolder.HADOOP_CONFIG


def get_hdfs():
    """Get hadoop FileSystem handler"""
    if not SparkConfigHolder.HDFS:
        SparkConfigHolder.HDFS = get_spark_context()._gateway.jvm.org.apache.hadoop.fs.FileSystem.get(
            get_hadoop_config()
        )

    return SparkConfigHolder.HDFS


def get_local_fs():
    """Get local FileSystem handler"""
    if not SparkConfigHolder.LOCAL_FS:
        SparkConfigHolder.LOCAL_FS = get_hdfs().getLocal(get_hadoop_config())

    return SparkConfigHolder.LOCAL_FS


def read_df(infile):
    """Read spark dataframe"""
    return (
        get_spark_session.read.format("csv")
        .option("header", "true")
        .option("inferschema", "true")
        .option("mode", "DROPMALFORMED")
        .option("mergeSchema", "true")
        .load(infile)
    )


def read_text_file(infile):
    """Read contents of text file from HDFS using Spark"""
    return "\n".join(
        [row["value"] for row in get_spark_session().read.format("text").load(infile).collect()]
    )


def read_json(infile):
    """Read JSON file as a dictionary from HDFS"""
    return json.loads(read_text_file(infile))


def read_yaml(infile):
    """Read YAML file as a dictionary from HDFS"""
    return yaml.safe_load(read_text_file(infile))


def copy_from_hdfs(src, dest, logger):
    """
    Copy a directory/file from HDFS to local filesystem

    Args:
        - src: String path to source(on HDFS)
        - dest: String path to destination(on local file system)
        - logger: logging handler
    """
    if logger:
        logger.info("Copying files from {} to {}".format(src, dest))

    get_hdfs().copyToLocalFile(get_path_from_str(src), get_path_from_str(dest))


def copy_to_hdfs(src, dest, overwrite=True, logger=None):
    """
    Copy a directory/file to HDFS from local filesystem

    Args:
        - src: String path to source(on local file system)
        - dest: String path to destination(on HDFS)
        - overwrite: Boolean to specify whether existing destination files should be overwritten
        - logger: logging handler
    """
    if logger:
        logger.info("Copying files from {} to {}".format(src, dest))

    if overwrite and get_hdfs().exists(get_path_from_str(dest)):
        get_hdfs().delete(get_path_from_str(dest), "true")

    get_hdfs().copyFromLocalFile(get_path_from_str(src), get_path_from_str(dest))
