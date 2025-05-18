import mlflow
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import IntegerType, StringType, BooleanType, FloatType, StructField, StructType


def main():
    schema = StructType([
        StructField("_c0", IntegerType(), True),
        StructField("id", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("target", IntegerType(), True),
        StructField("vote_count", IntegerType(), True),
        StructField("status", StringType(), True),
        StructField("release_date", StringType(), True),
        StructField("revenue", IntegerType(), True),
        StructField("runtime", IntegerType(), True),
        StructField("adult", BooleanType(), True),
        StructField("budget", IntegerType(), True),
        StructField("imdb_id", StringType(), True),
        StructField("original_language", StringType(), True),
        StructField("original_title", StringType(), True),
        StructField("overview", StringType(), True),
        StructField("popularity", FloatType(), True),
        StructField("tagline", StringType(), True),
        StructField("genres", StringType(), True),
        StructField("production_companies", StringType(), True),
        StructField("production_countries", StringType(), True),
        StructField("spoken_languages", StringType(), True),
        StructField("keywords", StringType(), True)
    ])

    builder = (
        SparkSession.builder.master("local[*]")
        .appName("DataIngestion")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .config("spark.hadoop.security.authentication", "simple")
    )

    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    mlflow.set_tracking_uri("file:/app/logs/mlruns")
    mlflow.set_experiment("MovieDataPipeline")

    print("Reading the raw CSV data and writing to the Bronze layer...")
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .option("quote", "\"") \
        .option("escape", "\"") \
        .csv("src/data/test_data.csv")

    df = df.drop("_c0")

    print(df.columns)
    df.show()

    df = df.repartition(4)

    if "target" in df.columns:
        df.write.format("delta") \
            .mode("overwrite") \
            .option("overwriteSchema", "true") \
            .partitionBy("target") \
            .save("./data/bronze/")
    else:
        df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").save("./data/bronze/")

    print("Data saved in the Bronze layer (./data/bronze/)")

    try:
        spark.sql("CREATE TABLE IF NOT EXISTS bronze_table USING DELTA LOCATION '/app/data/bronze/'")
        spark.sql("OPTIMIZE bronze_table")
        print("Bronze layer optimized.")
    except Exception as e:
        print(f"Could not optimize the Bronze layer: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
