import mlflow
import mlflow.spark
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from delta import configure_spark_with_delta_pip
from pyspark.sql.types import DoubleType, IntegerType


def main():
    builder = (
        SparkSession.builder.master("local[*]")
        .appName("Pipeline")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .config("spark.jars.packages", "io.delta:delta-core_2.12:2.0.0")
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .config("spark.hadoop.security.authentication", "simple")
    )
    spark = configure_spark_with_delta_pip(builder).getOrCreate()

    mlflow.set_tracking_uri("file:/app/logs/mlruns")
    mlflow.set_experiment("Default")

    print("Readeing data from Silver layer...")
    df = spark.read.format("delta").load("./data/silver/")
    df.show()

    print("Schema for Silver layer:")
    df.printSchema()

    print("Aggregating data by target...")
    agg_df = df.groupBy("target").count().orderBy("count", ascending=False)
    agg_df.show()

    df = df.dropna(subset=["target"])

    target_indexer = StringIndexer(inputCol="target", outputCol="target_index")
    df = target_indexer.fit(df).transform(df)

    df = df.withColumn("id", df["id"].cast(IntegerType()))
    df = df.withColumn("vote_count", df["vote_count"].cast(IntegerType()))
    df = df.withColumn("revenue", df["revenue"].cast(DoubleType()))
    df = df.withColumn("runtime", df["runtime"].cast(DoubleType()))
    df = df.withColumn("budget", df["budget"].cast(DoubleType()))
    df = df.withColumn("popularity", df["popularity"].cast(DoubleType()))

    feature_cols = [c for c in df.columns if c not in ["target", "target_index"]]

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")

    lr = LogisticRegression(featuresCol="features", labelCol="target_index")

    pipeline = Pipeline(stages=[assembler, lr])

    data_train, data_test = df.randomSplit([0.75, 0.25], seed=42)

    input_example_spark = data_test.select(*feature_cols).limit(1)
    input_example_pandas = input_example_spark.toPandas()

    with mlflow.start_run(run_name="LogisticRegression_Pipeline"):
        print("Training the model...")
        model = pipeline.fit(data_train)

        mlflow.spark.log_model(
            model,
            "logistic_regression_pipeline_model",
            input_example=input_example_pandas
        )
        mlflow.log_param("maxIter", lr.getMaxIter())
        mlflow.log_param("regParam", lr.getRegParam())

        training_summary = model.stages[-1].summary
        if training_summary:
            accuracy = training_summary.accuracy
            f1 = training_summary.fMeasureByLabel()[1] if len(training_summary.fMeasureByLabel()) > 1 else 0
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1", f1)
            print(f"Model trained. Accuracy: {accuracy}, F1: {f1}")
        else:
            print("No training summary available.")

    spark.stop()


if __name__ == "__main__":
    main()
