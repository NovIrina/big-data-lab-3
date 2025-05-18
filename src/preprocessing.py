import mlflow
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col, to_date, year
from delta import configure_spark_with_delta_pip


def remove_nan_and_duplicates(df):
    return df.dropna().dropDuplicates()


def remove_unnecessary_columns(df):
    for unwanted in df.columns:
        if unwanted == "_c0" or unwanted.lower().startswith("unnamed") or unwanted.strip() == "":
            df = df.drop(unwanted)
            print(f"Removed column: {unwanted}")

    columns_to_drop = [
        "title", "imdb_id", "original_language", "original_title",
        "overview", "tagline", "genres", "production_companies",
        "production_countries", "spoken_languages", "keywords"
    ]
    return df.drop(*columns_to_drop)


def convert_features(df):
    df = df.withColumn("adult", when(col("adult") == "True", 1).otherwise(0))
    df = df.withColumn("release_date", to_date(col("release_date"), "yyyy-MM-dd"))
    df = df.dropna(subset=["release_date"])
    df = df.withColumn("release_year", year(col("release_date"))).drop("release_date")
    return df


def filter_released_and_drop_status(df):
    df = df.filter(col("status") == "Released")
    return df.drop("status")


def main():
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("Preprocessing")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .config("spark.hadoop.security.authentication", "simple")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    mlflow.set_tracking_uri("file:/app/logs/mlruns")

    print("Reading the Bronze layer...")
    df = spark.read.format("delta").load("./data/bronze/")
    df.show()

    df = remove_unnecessary_columns(df)
    df = remove_nan_and_duplicates(df)
    df = convert_features(df)
    df = filter_released_and_drop_status(df)

    df = df.repartition(4)
    df.write.format("delta").mode("overwrite").option("overwriteSchema", "true") \
            .partitionBy("target").save("./data/silver/")
    print("Data written to Silver layer: ./data/silver/")

    try:
        spark.sql("CREATE TABLE IF NOT EXISTS silver_table USING DELTA LOCATION '/app/data/silver/'")
        spark.sql("OPTIMIZE silver_table ZORDER BY (release_year)")
        print("Optimization of Silver layer completed.")
    except Exception as e:
        print(f"Could not optimize Silver layer: {e}")

    spark.stop()


if __name__ == "__main__":
    main()
