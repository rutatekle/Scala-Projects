import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.Tokenizer
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.feature.HashingTF
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.{SaveMode, SparkSession}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder



object TweetSentiment {
  def main(args: Array[String]): Unit = {
    if (args.length != 2) {
      println("Usage: TweetSentiment InputDir OutputDir ")

    }


    val input_file = "s3a://bucket-bigdata-aws/Tweets.csv"
    val output_file = "s3a://bucket-bigdata-aws/tweetResult"
//    val input_file = args(0)
//    val output_file = args(1)


    val spark: SparkSession = SparkSession.builder()
      .master("local[1]")
      .appName("Tweets_Sentiment_Classifier")
      .getOrCreate()


    spark.sparkContext.hadoopConfiguration.set("fs.s3a.access.key", "AKIAVE4VFGETRWHHZDGS")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.secret.key", "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")
    spark.sparkContext.hadoopConfiguration.set("fs.s3a.endpoint", "s3.amazonaws.com")



    val input = spark.read.option("header","true").option("inferSchema","true").csv(input_file)
    val tweets = input.na.drop(Array ("text")).select("airline_sentiment","text")


    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("allWords")
    val remover = new StopWordsRemover().setInputCol(tokenizer.getOutputCol).setOutputCol("words")
    val indexer = new StringIndexer().setInputCol("airline_sentiment").setOutputCol("label")
    val hashingTf = new HashingTF().setInputCol(remover.getOutputCol).setOutputCol("features")
    val lr = new LogisticRegression().setMaxIter(10).setRegParam(0.001).setLabelCol(indexer.getOutputCol)


    val pipeline = new Pipeline().setStages(Array(tokenizer, remover,indexer, hashingTf, lr))


    val param = new ParamGridBuilder()
      .addGrid(hashingTf.numFeatures, Array(1000,10000,100000))
      .addGrid(lr.regParam, Array(0.1,0.01, 0.001))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(param)
      .setNumFolds(2)


    val model = crossValidator.fit(tweets)
    val result = model.transform(tweets)
    val predictionResult = result.select("label","prediction")


    predictionResult.write.mode(SaveMode.Overwrite).csv(output_file)

    evaluator.setLabelCol("label")
    evaluator.setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictionResult)

    println("Accuracy = "+"%6.3f".format(accuracy))


  }
}